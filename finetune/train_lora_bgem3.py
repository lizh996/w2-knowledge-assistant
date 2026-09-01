# -*- coding: utf-8 -*-
"""
BGE-M3 Dense Fine-tuning (LoRA) — W3 微调 E1 实验
方法: FlagEmbedding(BGE-M3) 模型 + peft LoRA + InfoNCE 对比学习
数据: jsonl, 每行 {"query": "...", "pos": ["..."], "neg": ["..."]}
用法: D:\\an\\envs\\mineru\\python.exe train_lora_bgem3.py
"""
import json, torch, random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model

# ============ 参数（讲师标准，按需改） ============
MODEL_PATH = r"C:\Users\lizhihao\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"
import os as _os
BASE = _os.path.dirname(_os.path.abspath(__file__))
TRAIN_DATA = _os.path.join(BASE, "train.jsonl")  # 训练数据（800 条）
OUTPUT_DIR = _os.path.join(BASE, "output_ft_v1")  # LoRA 保存目录
LORA_R, LORA_ALPHA, LORA_DROPOUT = 8, 16, 0.05
TARGETS = ["query", "value"]  # 匹配 encoder.layer.*.attention.self.query/value
LR = 1e-4
EPOCHS = 3
BATCH = 2          # 8GB 显存限制; 配合 GRAD_ACC=4 等效 batch 8
QUERY_MAX, PASSAGE_MAX = 256, 512   # 讲师标准
TEMPERATURE = 0.05                  # 讲师标准 0.02-0.05
GRAD_ACC = 4        # 梯度累积: 2x4=8(等效batch, 讲师标准)
SEED = 42
random.seed(SEED); torch.manual_seed(SEED)

# ============ 数据 ============
class PairDataset(Dataset):
    def __init__(self, path, tok):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        self.tok = tok
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        q = self.tok(r["query"], max_length=QUERY_MAX, truncation=True, padding="max_length", return_tensors="pt")
        p = self.tok(random.choice(r["pos"]), max_length=PASSAGE_MAX, truncation=True, padding="max_length", return_tensors="pt")
        n = self.tok(random.choice(r["neg"]), max_length=PASSAGE_MAX, truncation=True, padding="max_length", return_tensors="pt")
        return q["input_ids"][0], q["attention_mask"][0], p["input_ids"][0], p["attention_mask"][0], n["input_ids"][0], n["attention_mask"][0]

device = "cuda" if torch.cuda.is_available() else "cpu"  # 模块级设备

def embed(ids, mask):
    """取 [CLS] 向量 + L2 归一化（BGE-M3 dense 表示）"""
    ids, mask = ids.to(device), mask.to(device)
    h = model(input_ids=ids, attention_mask=mask).last_hidden_state
    v = h[:, 0]
    return torch.nn.functional.normalize(v, p=2, dim=-1)

def info_nce(q, pos, neg, temp=TEMPERATURE):
    """InfoNCE: 拉近 q-pos, 推远 q-neg + batch 内其他 query（in-batch neg, 排除自身）"""
    sim_pos = (q * pos).sum(-1) / temp                       # [B]
    sim_neg = (q * neg).sum(-1) / temp                       # [B]
    sim_q   = (q @ q.T) / temp                               # [B,B] in-batch
    diag = torch.eye(q.size(0), device=q.device, dtype=torch.bool)
    sim_q = sim_q.masked_fill(diag, float("-inf"))           # 排除自己（不是负例）
    cand = torch.cat([sim_pos.unsqueeze(1), sim_neg.unsqueeze(1), sim_q], dim=1)
    return -sim_pos + torch.logsumexp(cand, dim=1)           # 即 InfoNCE

# ============ 主流程 ============
if __name__ == "__main__":
    print(f"训练设备: {device}")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    base = AutoModel.from_pretrained(MODEL_PATH)
    # peft 注入 LoRA（query/value 投影）
    lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                      target_modules=TARGETS, task_type="FEATURE_EXTRACTION")
    model = get_peft_model(base, lora)
    model = model.to(device)
    model.train()
    print("可训练参数: {:,} / 总参数: {:,}".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        sum(p.numel() for p in model.parameters())))

    dl = DataLoader(PairDataset(TRAIN_DATA, tok), batch_size=BATCH, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * len(dl))

    total_steps = 0
    for ep in range(EPOCHS):
        loss_sum, n = 0.0, 0
        for qi, qm, pi, pm, ni, nm in dl:
            qi, qm = qi.cuda(), qm.cuda()
            pi, pm, ni, nm = pi.cuda(), pm.cuda(), ni.cuda(), nm.cuda()
            q, p, neg = embed(qi, qm), embed(pi, pm), embed(ni, nm)
            loss = info_nce(q, p, neg).mean() / GRAD_ACC
            loss.backward()
            if total_steps % GRAD_ACC == 0:
                opt.step(); opt.zero_grad(); sched.step()
            loss_sum += loss.item(); n += 1; total_steps += 1
            if total_steps % 10 == 0:
                print(f"  ep{ep+1} step{total_steps} loss={loss.item()*GRAD_ACC:.4f}")
        print(f"Epoch {ep+1} 平均 loss = {loss_sum/n:.4f}")

    model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print(f"\\n✅ LoRA 已保存: {OUTPUT_DIR}（推理时 base 模型 + adapter 合并，原模型未动）")
