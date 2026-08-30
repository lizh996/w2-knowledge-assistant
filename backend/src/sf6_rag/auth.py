"""认证：登录 + token + 角色（admin/user）。

教学级实现：单进程内存 token，不做数据库/刷新/JWT。生产应替换为
JWT + 密码哈希（bcrypt）+ 持久化会话。

用法：
    from sf6_rag.auth import login, get_current_user
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass

# 用户表（教学演示，明文密码；生产必须 bcrypt 哈希 + 环境变量/数据库）
_USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "user": {"password": "user123", "role": "user"},
}

# token → username（单进程内存；重启失效）
_TOKENS: dict[str, str] = {}


@dataclass
class User:
    username: str
    role: str


def _make_token() -> str:
    return secrets.token_urlsafe(32)


def login(username: str, password: str) -> dict | None:
    """校验用户名密码，成功返回 {token, role}，失败返回 None。"""
    user = _USERS.get(username)
    if user is None:
        return None
    if not secrets.compare_digest(user["password"], password):
        return None
    token = _make_token()
    _TOKENS[token] = username
    return {"token": token, "role": user["role"]}


def verify_token(token: str) -> User | None:
    """校验 token，返回 User；无效/过期返回 None。"""
    username = _TOKENS.get(token)
    if username is None:
        return None
    user = _USERS.get(username)
    if user is None:
        return None
    return User(username=username, role=user["role"])


def get_current_user(authorization: str | None) -> User | None:
    """从 Authorization: Bearer <token> 头解析当前用户。"""
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return verify_token(parts[1])
