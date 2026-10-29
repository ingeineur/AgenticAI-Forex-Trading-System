"""FastAPI routes for strategy management."""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from .auth import login, validate, logout
from .process import (get_process, list_processes, create_process,
                       delete_process, save_config, load_configs)

router = APIRouter()


def require_auth(req: Request):
    auth = req.headers.get("Authorization", "")
    token = auth.replace("Bearer ", "").strip()
    if not validate(token):
        raise HTTPException(401, "Unauthorized")


# --- Auth ---

class LoginBody(BaseModel):
    password: str

@router.post("/auth/login")
def api_login(body: LoginBody):
    token = login(body.password)
    if not token:
        raise HTTPException(401, "Invalid password")
    return {"token": token}


# --- Strategies ---

@router.get("/strategies")
def api_list(req: Request):
    require_auth(req)
    load_configs()
    return list_processes()


@router.get("/strategies/{instance_id}")
def api_get(instance_id: str, req: Request):
    require_auth(req)
    p = get_process(instance_id)
    if not p:
        raise HTTPException(404, "Not found")
    return p.info()


class CreateBody(BaseModel):
    id: str
    file: str
    symbol: str
    tf: str = "h1"
    args: list = []

@router.post("/strategies")
def api_create(body: CreateBody, req: Request):
    require_auth(req)
    cfg = {
        "id": body.id, "file": body.file,
        "symbol": body.symbol, "tf": body.tf,
        "args": body.args,
    }
    p = create_process(body.id, cfg)
    if not p:
        raise HTTPException(409, f"Strategy '{body.id}' already exists")
    return p.info()


@router.post("/strategies/{instance_id}/start")
def api_start(instance_id: str, req: Request):
    require_auth(req)
    p = get_process(instance_id)
    if not p:
        raise HTTPException(404, "Not found")
    ok = p.start()
    if not ok:
        raise HTTPException(409, "Already running")
    return {"status": "started"}


@router.post("/strategies/{instance_id}/stop")
def api_stop(instance_id: str, req: Request):
    require_auth(req)
    p = get_process(instance_id)
    if not p:
        raise HTTPException(404, "Not found")
    p.stop()
    return {"status": "stopped"}


@router.post("/strategies/{instance_id}/restart")
def api_restart(instance_id: str, req: Request):
    require_auth(req)
    p = get_process(instance_id)
    if not p:
        raise HTTPException(404, "Not found")
    ok = p.restart()
    return {"status": "restarted" if ok else "failed"}


class UpdateBody(BaseModel):
    file: str = None
    symbol: str = None
    tf: str = None
    args: list = None

@router.put("/strategies/{instance_id}/config")
def api_update(instance_id: str, body: UpdateBody, req: Request):
    require_auth(req)
    p = get_process(instance_id)
    if not p:
        raise HTTPException(404, "Not found")
    # Stop process if running, update config, save
    was_running = p.status == "running"
    if was_running:
        p.stop()
    if body.file is not None: p.config["file"] = body.file
    if body.symbol is not None: p.config["symbol"] = body.symbol
    if body.tf is not None: p.config["tf"] = body.tf
    if body.args is not None: p.config["args"] = body.args
    save_config(instance_id, p.config)
    if was_running:
        p.start()
    return {"status": "updated", "info": p.info()}


@router.delete("/strategies/{instance_id}")
def api_delete(instance_id: str, req: Request):
    require_auth(req)
    delete_process(instance_id)
    return {"status": "deleted"}


@router.get("/strategies/{instance_id}/logs")
def api_logs(instance_id: str, n: int = 50, req: Request = None):
    require_auth(req)
    p = get_process(instance_id)
    if not p:
        raise HTTPException(404, "Not found")
    return {"lines": p.tail_log(n)}


_BT_MAP = {"BULLISH": 1, "SIDEWAYS": 0, "BEARISH": -1}

@router.get("/market/trends")
def api_market_trends(req: Request, offset: int = 0, d1: int = 0):
    require_auth(req)
    from .mt5_pnl import _ensure_mt5
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from trend import get_trend
    results = {}
    for sym in ("XAUUSD", "BTCUSD", "NAS100"):
        try:
            is_acc, dom_bias, tf_bias, rev_pct = get_trend(sym, "h1", use_d1=bool(d1), offset_hours=offset)
            results[sym] = {
                "trend": dom_bias,
                "tf_trend": tf_bias,
                "accumulating": is_acc,
                "reversal_pct": rev_pct,
            }
        except Exception as e:
            results[sym] = {"trend": None, "tf_trend": None, "accumulating": True, "error": str(e)}
    return results


@router.get("/positions/open")
def api_open_positions(req: Request):
    require_auth(req)
    from .mt5_pnl import _ensure_mt5
    if not _ensure_mt5():
        return {"count": 0, "total_pnl": 0}
    import MetaTrader5 as mt5
    positions = mt5.positions_get()
    if not positions:
        return {"count": 0, "total_pnl": 0}
    instance_ids = list_processes()
    short_ids = [p.get('short_id', '') for p in instance_ids if p.get('short_id')]
    count = 0
    total_pnl = 0.0
    for p in positions:
        if not p.comment: continue
        if any(sid in p.comment for sid in short_ids):
            count += 1
            total_pnl += p.profit
    return {"count": count, "total_pnl": round(total_pnl, 2)}


# --- Templates ---

TEMPLATES = [
    {"id": "mss_xau_h1", "file": "sweep_mss_zone.py", "symbol": "XAUUSD", "tf": "h1",
     "args": ["--market-only", "--sl-atr", "1.2", "--tp-atr", "0.5", "--orderflow",
              "--of-mode", "ohlc", "--lookback", "100", "--max-signal-age", "48", "--ml-threshold", "0.7", "--volume", "0.01"]},
    {"id": "mss_xau_m15", "file": "sweep_mss_zone.py", "symbol": "XAUUSD", "tf": "m15",
     "args": ["--market-only", "--sl-atr", "1.2", "--tp-atr", "0.5", "--orderflow",
              "--of-mode", "ohlc", "--lookback", "100", "--max-signal-age", "96", "--ml-threshold", "0.7", "--volume", "0.01"]},
    {"id": "smc_xau_h1", "file": "smc_strategy.py", "symbol": "XAUUSD", "tf": "h1",
     "args": ["--market-only", "--entry-mode", "all", "--sl-atr", "1.2", "--tp-atr", "0.5",
              "--orderflow", "--of-mode", "ohlc", "--lookback", "200", "--max-signal-age", "48", "--ml-threshold", "0.7", "--volume", "0.01"]},
    {"id": "smc_btc_h1", "file": "smc_strategy.py", "symbol": "BTCUSD", "tf": "h1",
     "args": ["--market-only", "--entry-mode", "all", "--sl-atr", "1.2", "--tp-atr", "0.5",
              "--orderflow", "--of-mode", "ohlc", "--lookback", "200", "--max-signal-age", "48", "--ml-threshold", "0.8", "--volume", "0.01"]},
    {"id": "mss_xau_m15_v2", "file": "sweep_mss_zone_v2.py", "symbol": "XAUUSD", "tf": "m15",
     "args": ["--market-only", "--skip-swept", "--ob-sweep", "--sl-atr", "2.0", "--tp-atr", "1.0",
              "--orderflow", "--of-mode", "ohlc", "--lookback", "200", "--max-signal-age", "96",
              "--ml-threshold", "0.6", "--max-orders", "2", "--volume", "0.01"]},
    {"id": "mss_btc_m15_v2", "file": "sweep_mss_zone_v2.py", "symbol": "BTCUSD", "tf": "m15",
     "args": ["--market-only", "--skip-swept", "--sl-atr", "2.0", "--tp-atr", "1.0",
              "--orderflow", "--of-mode", "ohlc", "--lookback", "200", "--max-signal-age", "96",
              "--ml-threshold", "0.6", "--max-orders", "2", "--volume", "0.02"]},
    {"id": "mss_xau_h1_v2", "file": "sweep_mss_zone_v2.py", "symbol": "XAUUSD", "tf": "h1",
     "args": ["--market-only", "--skip-swept", "--ob-sweep", "--sl-atr", "2.0", "--tp-atr", "1.0",
              "--orderflow", "--of-mode", "ohlc", "--lookback", "100", "--max-signal-age", "24",
              "--ml-threshold", "0.6", "--max-orders", "2", "--volume", "0.01"]},
    {"id": "mss_btc_h1_v2", "file": "sweep_mss_zone_v2.py", "symbol": "BTCUSD", "tf": "h1",
     "args": ["--market-only", "--skip-swept", "--sl-atr", "3.0", "--tp-atr", "1.0",
              "--orderflow", "--of-mode", "ohlc", "--lookback", "100", "--max-signal-age", "24",
              "--ml-threshold", "0.6", "--max-orders", "2", "--volume", "0.02"]},
    {"id": "mss_xau_m15_v3", "file": "sweep_mss_zone_v3.py", "symbol": "XAUUSD", "tf": "m15",
     "args": ["--market-only", "--skip-swept", "--ob-sweep", "--sl-atr", "2.0", "--tp-atr", "1.0",
              "--orderflow", "--of-mode", "ohlc", "--lookback", "200", "--max-signal-age", "96",
              "--ml-threshold", "0.7", "--max-orders", "2", "--volume", "0.01"]},
    {"id": "mss_btc_m15_v3", "file": "sweep_mss_zone_v3.py", "symbol": "BTCUSD", "tf": "m15",
     "args": ["--market-only", "--skip-swept", "--sl-atr", "2.0", "--tp-atr", "1.0",
              "--orderflow", "--of-mode", "ohlc", "--lookback", "200", "--max-signal-age", "96",
              "--ml-threshold", "0.7", "--max-orders", "2", "--volume", "0.02"]},
    {"id": "mss_xau_h1_v3", "file": "sweep_mss_zone_v3.py", "symbol": "XAUUSD", "tf": "h1",
     "args": ["--market-only", "--skip-swept", "--ob-sweep", "--sl-atr", "2.0", "--tp-atr", "1.0",
              "--orderflow", "--of-mode", "ohlc", "--lookback", "100", "--max-signal-age", "24",
              "--ml-threshold", "0.7", "--max-orders", "2", "--volume", "0.01"]},
    {"id": "mss_btc_h1_v3", "file": "sweep_mss_zone_v3.py", "symbol": "BTCUSD", "tf": "h1",
     "args": ["--market-only", "--skip-swept", "--sl-atr", "3.0", "--tp-atr", "1.0",
              "--orderflow", "--of-mode", "ohlc", "--lookback", "100", "--max-signal-age", "24",
              "--ml-threshold", "0.7", "--max-orders", "2", "--volume", "0.02"]},
    {"id": "mss_xau_m15_v4", "file": "sweep_mss_zone_v4.py", "symbol": "XAUUSD", "tf": "m15",
     "args": ["--market-only", "--skip-swept", "--ob-sweep", "--sl-atr", "2.0", "--tp-atr", "1.0",
              "--orderflow", "--of-mode", "ohlc", "--lookback", "200", "--max-signal-age", "96",
              "--ml-threshold", "0.7", "--max-orders", "2", "--volume", "0.01"]},
    {"id": "mss_btc_m15_v4", "file": "sweep_mss_zone_v4.py", "symbol": "BTCUSD", "tf": "m15",
     "args": ["--market-only", "--skip-swept", "--sl-atr", "2.0", "--tp-atr", "1.0",
              "--orderflow", "--of-mode", "ohlc", "--lookback", "200", "--max-signal-age", "96",
              "--ml-threshold", "0.7", "--max-orders", "2", "--volume", "0.02"]},
    {"id": "mss_xau_h1_v4", "file": "sweep_mss_zone_v4.py", "symbol": "XAUUSD", "tf": "h1",
     "args": ["--market-only", "--skip-swept", "--ob-sweep", "--sl-atr", "2.0", "--tp-atr", "1.0",
              "--orderflow", "--of-mode", "ohlc", "--lookback", "100", "--max-signal-age", "24",
              "--ml-threshold", "0.7", "--max-orders", "2", "--volume", "0.01"]},
    {"id": "mss_btc_h1_v4", "file": "sweep_mss_zone_v4.py", "symbol": "BTCUSD", "tf": "h1",
     "args": ["--market-only", "--skip-swept", "--sl-atr", "3.0", "--tp-atr", "1.0",
              "--orderflow", "--of-mode", "ohlc", "--lookback", "100", "--max-signal-age", "24",
              "--ml-threshold", "0.7", "--max-orders", "2", "--volume", "0.02"]},
    {"id": "mss_nas100_m15_v1", "file": "sweep_mss_zone.py", "symbol": "NAS100", "tf": "m15",
     "args": ["--market-only", "--skip-swept", "--ob-sweep", "--sl-atr", "2.0", "--tp-atr", "1.0",
              "--orderflow", "--of-mode", "ohlc", "--lookback", "200", "--max-signal-age", "96",
              "--ml-threshold", "0.7", "--max-orders", "2", "--volume", "0.1"]},
    {"id": "mss_nas100_m15_v2", "file": "sweep_mss_zone_v2.py", "symbol": "NAS100", "tf": "m15",
     "args": ["--market-only", "--skip-swept", "--ob-sweep", "--sl-atr", "2.0", "--tp-atr", "1.0",
              "--orderflow", "--of-mode", "ohlc", "--lookback", "200", "--max-signal-age", "96",
              "--ml-threshold", "0.7", "--max-orders", "2", "--volume", "0.1"]},
    {"id": "mss_nas100_m15_v3", "file": "sweep_mss_zone_v3.py", "symbol": "NAS100", "tf": "m15",
     "args": ["--market-only", "--skip-swept", "--ob-sweep", "--sl-atr", "2.0", "--tp-atr", "1.0",
              "--orderflow", "--of-mode", "ohlc", "--lookback", "200", "--max-signal-age", "96",
              "--ml-threshold", "0.7", "--max-orders", "2", "--volume", "0.1"]},
    {"id": "mss_nas100_m15_v4", "file": "sweep_mss_zone_v4.py", "symbol": "NAS100", "tf": "m15",
     "args": ["--market-only", "--skip-swept", "--ob-sweep", "--sl-atr", "2.0", "--tp-atr", "1.0",
              "--orderflow", "--of-mode", "ohlc", "--lookback", "200", "--max-signal-age", "96",
              "--ml-threshold", "0.7", "--max-orders", "2", "--volume", "0.1"]},
    {"id": "mss_nas100_h1_v1", "file": "sweep_mss_zone.py", "symbol": "NAS100", "tf": "h1",
     "args": ["--market-only", "--skip-swept", "--ob-sweep", "--sl-atr", "2.0", "--tp-atr", "1.0",
              "--orderflow", "--of-mode", "ohlc", "--lookback", "100", "--max-signal-age", "24",
              "--ml-threshold", "0.6", "--max-orders", "2", "--volume", "0.1"]},
    {"id": "mss_nas100_h1_v2", "file": "sweep_mss_zone_v2.py", "symbol": "NAS100", "tf": "h1",
     "args": ["--market-only", "--skip-swept", "--ob-sweep", "--sl-atr", "2.0", "--tp-atr", "1.0",
              "--orderflow", "--of-mode", "ohlc", "--lookback", "100", "--max-signal-age", "24",
              "--ml-threshold", "0.6", "--max-orders", "2", "--volume", "0.1"]},
    {"id": "mss_nas100_h1_v3", "file": "sweep_mss_zone_v3.py", "symbol": "NAS100", "tf": "h1",
     "args": ["--market-only", "--skip-swept", "--ob-sweep", "--sl-atr", "2.0", "--tp-atr", "1.0",
              "--orderflow", "--of-mode", "ohlc", "--lookback", "100", "--max-signal-age", "24",
              "--ml-threshold", "0.6", "--max-orders", "2", "--volume", "0.1"]},
    {"id": "mss_nas100_h1_v4", "file": "sweep_mss_zone_v4.py", "symbol": "NAS100", "tf": "h1",
     "args": ["--market-only", "--skip-swept", "--ob-sweep", "--sl-atr", "2.0", "--tp-atr", "1.0",
              "--orderflow", "--of-mode", "ohlc", "--lookback", "100", "--max-signal-age", "24",
              "--ml-threshold", "0.6", "--max-orders", "2", "--volume", "0.1"]},
]

@router.get("/templates")
def api_templates(req: Request):
    require_auth(req)
    return TEMPLATES
