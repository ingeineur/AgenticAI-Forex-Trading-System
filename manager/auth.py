"""Simple password auth — no JWT, just session tokens."""
import secrets
import os

_SECRET_FILE = os.path.join(os.path.dirname(__file__), '.secret')
_tokens = {}  # token -> username (stored in memory)

def _load_password():
    if os.path.exists(_SECRET_FILE):
        with open(_SECRET_FILE, 'r') as f:
            return f.read().strip()
    pwd = "password123"
    with open(_SECRET_FILE, 'w') as f:
        f.write(pwd)
    return pwd

PASSWORD = _load_password()

def login(password: str):
    if password == PASSWORD:
        token = secrets.token_hex(32)
        _tokens[token] = "admin"
        return token
    return None

def validate(token: str):
    return token in _tokens

def logout(token: str):
    _tokens.pop(token, None)
