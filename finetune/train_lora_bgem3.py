# -*- coding: utf-8 -*-
"""
BGE-M3 Dense Fine-tuning (LoRA) — E2 实验（E1 修复版）
相较 E1 的修复:
  P1-3 InfoNCE in-batch 负例改用 q×passage(其他样本的 positive)，不再用 q×query
  P1-4 梯度累积计数修正 ((step+1) % GRAD_ACC == 0，避免首个 batch 即 step)
  P1-5 T_max 对齐 optimizer 实际 step 次数 (总步数 // GRAD_ACC)
  P1-7 pos/neg 同长 512 (数据侧 gen_data 已对齐，训练 PASSAGE_MAX=512 一致)
  新增: 独立输出目录 output_ft_v2 + 每 epoch checkpoint + 按 chunk 划分验证集 + 早停
数据: train_clean.jsonl (gen_hard_neg.py 产出: query 去重 + base 模型检索的 hard negative)
用法: D:\\an\\envs\\mineru\\python.exe train_lora_bgem3.py
"""
import json, os, random, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model

# ============ 参数（讲师标准，按需改） ============
MODEL_PATH = r"C:\Users\lizhihao\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"
BASE = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA = os.path.join(BASE, "train_clean.jsonl")  # hard neg + 去重后的干净数据
OUTPUT_DIR = os.path.join(BASE, "output_ft_v2")       # 独立输出目录 (E1 用 output_ft_v1)
LORA_R, LORA_ALPHA, LORA_DROPOUT = 8, 16, 0.05
TARGETS = ["query", "value"]  # 匹配 encoder.layer.*.attention.self.query/value
LR = 1e-4
EPOCHS = 3
BATCH = 2          # 8GB 显存限制; 配合 GRAD_ACC=4 等效 batch 8
QUERY_MAX, PASSAGE_MAX = 256, 512
TEMPERATURE = 0.05                  # 讲师标准 0.02-0.05
GRAD_ACC = 4        # 梯度累积: 2x4=8(等效batch, 讲师标准)
SEED = 42
VAL_FRAC = 0.1      # 按 chunk 划分验证集 (约 10% chunk，避免同 chunk 泄漏)
EARLY_STOP_PATIENCE = 2  # 验证集 loss 连续 N epoch 无改善则早停
random.seed(SEED); torch.manual_seed(SEED)

# ============ 数据 ============
def _load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

def split_by_chunk(rows, val_frac=VAL_FRAC):
    """按 chunk 划分训练/验证，避免同一 chunk 的 query 同时出现在两侧造成泄漏。"""
    chunks = sorted({r["knowledge_id"] for r in rows})
    random.shuffle(chunks)
    n_val = max(1, int(len(chunks) * val_frac))
    val_chunks = set(chunks[:n_val])
    train = [r for r in rows if r["knowledge_id"] not in val_chunks]
    val = [r for r in rows if r["knowledge_id"] in val_chunks]
    return train, val

class PairDataset(Dataset):
    def __init__(self, rows, tok):
        self.rows = rows
        self.tok = tok
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        out = self._enc(r["query"], QUERY_MAX) + self._enc(r["pos"][0], PASSAGE_MAX)
        negs = [self._enc(n, PASSAGE_MAX) for n in r["neg"][:2]]
        while len(negs) < 2:                       # 兜底: neg 不足 2 个则复用最后一个
            negs.append(negs[-1])
        for n in negs:
            out += n
        return out  # (q_ids, q_mask, p_ids, p_mask, n1_ids, n1_mask, n2_ids, n2_mask)
    def _enc(self, text, max_len):
        e = self.tok(text, max_length=max_len, truncation=True, padding="max_length", return_tensors="pt")
        return e["input_ids"][0], e["attention_mask"][0]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None  # 模块级, embed()/evaluate() 使用

def embed(ids, mask):
    """取 [CLS] 向量 + L2 归一化（BGE-M3 dense 表示）"""
    ids, mask = ids.to(device), mask.to(device)
    h = model(input_ids=ids, attention_mask=mask).last_hidden_state
    v = h[:, 0]
    return torch.nn.functional.normalize(v, p=2, dim=-1)

def info_nce(q, pos, negs, temp=TEMPERATURE):
    """InfoNCE: 正例=自身 pos; 负例=显式 hard neg(K 个) + in-batch(q × 其他样本的 pos)。"""
    B = q.size(0)
    sim_pos = (q * pos).sum(-1) / temp                    # [B]
    sim_negs = (q.unsqueeze(1) * negs).sum(-1) / temp     # [B, K]
    sim_qp = (q @ pos.T) / temp                           # [B, B] in-batch negative
    diag = torch.eye(B, device=q.device, dtype=torch.bool)
    sim_qp = sim_qp.masked_fill(diag, float("-inf"))      # 排除自身 pos (不是负例)
    cand = torch.cat([sim_pos.unsqueeze(1), sim_negs, sim_qp], dim=1)  # [B, 1+K+B]
    return -sim_pos + torch.logsumexp(cand, dim=1)

def unpack(batch):
    qi, qm, pi, pm, n1i, n1m, n2i, n2m = [b.to(device) for b in batch]
    q = embed(qi, qm); p = embed(pi, pm)
    negs = torch.stack([embed(n1i, n1m), embed(n2i, n2m)], dim=1)  # [B, 2, D]
    return q, p, negs

@torch.no_grad()
def evaluate(dl):
    model.eval()
    total, n = 0.0, 0
    for batch in dl:
        q, p, negs = unpack(batch)
        total += info_nce(q, p, negs).sum().item()
        n += q.size(0)
    model.train()
    return total / n

# ============ 主流程 ============
if __name__ == "__main__":
    print(f"训练设备: {device}")
    rows = _load_rows(TRAIN_DATA)
    train_rows, val_rows = split_by_chunk(rows)
    print(f"数据: 总 {len(rows)} 条 → 训练 {len(train_rows)} / 验证 {len(val_rows)} (按 chunk 划分)")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    base = AutoModel.from_pretrained(MODEL_PATH)
    lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                      target_modules=TARGETS, task_type="FEATURE_EXTRACTION")
    model = get_peft_model(base, lora)
    model = model.to(device)
    model.train()
    print("可训练参数: {:,} / 总参数: {:,}".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        sum(p.numel() for p in model.parameters())))

    dl = DataLoader(PairDataset(train_rows, tok), batch_size=BATCH, shuffle=True)
    dl_val = DataLoader(PairDataset(val_rows, tok), batch_size=BATCH, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999))
    # T_max 对齐 optimizer 实际 step 次数（每 GRAD_ACC 个 batch step 一次）
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * (len(dl) // GRAD_ACC))

    step = 0               # 累计 batch 步（用于梯度累积）
    best_val = float("inf")
    bad_epochs = 0
    for ep in range(EPOCHS):
        loss_sum, n = 0.0, 0
        for batch in dl:
            q, p, negs = unpack(batch)
            loss = info_nce(q, p, negs).mean() / GRAD_ACC
            loss.backward()
            step += 1
            if step % GRAD_ACC == 0:          # 累积满 GRAD_ACC 个 batch 才更新
                opt.step(); opt.zero_grad(); sched.step()
            loss_sum += loss.item() * GRAD_ACC; n += 1
            if step % 10 == 0:
                print(f"  ep{ep+1} step{step} loss={loss.item()*GRAD_ACC:.4f} lr={sched.get_last_lr()[0]:.2e}")
        train_loss = loss_sum / n
        val_loss = evaluate(dl_val)
        print(f"Epoch {ep+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        # checkpoint: 每 epoch 存一份
        ckpt_dir = os.path.join(OUTPUT_DIR, f"epoch_{ep+1}")
        model.save_pretrained(ckpt_dir); tok.save_pretrained(ckpt_dir)

        # best 保存 + 早停
        if val_loss < best_val - 1e-4:
            best_val = val_loss; bad_epochs = 0
            model.save_pretrained(OUTPUT_DIR); tok.save_pretrained(OUTPUT_DIR)
            print(f"  [best] val_loss={best_val:.4f} 已存 {OUTPUT_DIR}")
        else:
            bad_epochs += 1
            if bad_epochs >= EARLY_STOP_PATIENCE:
                print(f"[early-stop] 连续 {bad_epochs} epoch 无改善，停止训练")
                break

    print(f"\n[OK] 训练完成。best val_loss={best_val:.4f}，LoRA 已存: {OUTPUT_DIR}")
