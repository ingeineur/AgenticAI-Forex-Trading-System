"""Query MT5 for PnL by strategy instance ID (short_id in comment)."""
from datetime import datetime, timedelta
import os

try:
    import MetaTrader5 as mt5
    _HAS_MT5 = True
except ImportError:
    _HAS_MT5 = False

_DEFAULT_ACCOUNT = {"login": int(os.getenv("MT5_ACCOUNT", "0")), "password": os.getenv("MT5_PASSWORD", ""), "server": "PepperstoneBS-MT5-Live01"}
_initialized = False

# Point value per $1 risk for 0.01 lot
_POINT_VALUES = {"XAUUSD": 0.01, "BTCUSD": 0.01}  # $ per point for 0.01 lot


def _ensure_mt5():
    global _initialized
    if not _HAS_MT5:
        return False
    if not _initialized:
        try:
            if not mt5.initialize():
                return False
            ac = _DEFAULT_ACCOUNT
            if not mt5.login(ac["login"], ac["password"], ac["server"]):
                mt5.shutdown()
                return False
            _initialized = True
        except:
            return False
    return True


def _one_r_from_position(pos):
    """Estimate 1R (risk amount) from a position's SL distance and volume."""
    try:
        entry = pos.price_open
        sl = pos.sl
        vol = pos.volume
        symbol = pos.symbol
        if sl is None or sl <= 0 or entry is None or entry <= 0:
            return 10.0  # fallback
        sl_dist = abs(entry - sl)
        pv = _POINT_VALUES.get(symbol, 0.01)
        return max(sl_dist * pv * (vol / 0.01), 1.0)
    except:
        return 10.0


def get_pnl(short_id, days=7):
    """Get PnL in +/- R for a given instance short_id within last N days."""
    try:
        return _get_pnl(short_id, days)
    except Exception as e:
        return {"error": str(e), "trades": 0, "pnl_r": 0, "avg_r": 0, "wins": 0, "losses": 0}


def _get_pnl(short_id, days=7):
    if not _ensure_mt5():
        return {"error": "MT5 not available", "trades": 0, "pnl_r": 0, "avg_r": 0, "wins": 0, "losses": 0}

    from_date = datetime.now() - timedelta(days=days)
    to_date = datetime.now()

    # --- Closed positions (history deals) ---
    all_deals = mt5.history_deals_get(from_date, to_date)
    match_pids = set()
    if all_deals:
        for d in all_deals:
            if d.comment and short_id in d.comment and d.position_id > 0:
                match_pids.add(d.position_id)

    # Aggregate PnL per position from deals
    pos_pnl = {}
    pos_vol = {}
    pos_symbol = {}
    if all_deals:
        for d in all_deals:
            if d.position_id in match_pids:
                pos_pnl[d.position_id] = pos_pnl.get(d.position_id, 0) + d.profit
                if d.position_id not in pos_vol and d.volume > 0:
                    pos_vol[d.position_id] = d.volume
                    pos_symbol[d.position_id] = d.symbol

    # --- Open positions (floating PnL) ---
    open_positions = mt5.positions_get()
    open_pnl = {}
    if open_positions:
        for p in open_positions:
            if p.comment and short_id in p.comment:
                pid = p.ticket
                # Include floating PnL
                pos_pnl[pid] = pos_pnl.get(pid, 0) + p.profit
                if pid not in pos_vol:
                    pos_vol[pid] = p.volume
                    pos_symbol[pid] = p.symbol
                open_pnl[pid] = p

    # --- Special case: if no positions found but strategy is running, check open trades ---
    if not match_pids and open_positions:
        for p in open_positions:
            if p.comment and short_id in p.comment:
                match_pids.add(p.ticket)
                pos_pnl[p.ticket] = p.profit
                pos_vol[p.ticket] = p.volume
                pos_symbol[p.ticket] = p.symbol
                open_pnl[p.ticket] = p

    if not match_pids:
        return {"error": None, "trades": 0, "pnl_r": 0, "avg_r": 0, "wins": 0, "losses": 0}

    # Calculate R-multiples
    total_r = 0.0; wins = 0; losses = 0; breaks = 0

    for pid in match_pids:
        pnl = pos_pnl.get(pid, 0)
        vol = pos_vol.get(pid, 0.01)
        sym = pos_symbol.get(pid, "XAUUSD")
        one_r = 10.0  # default fallback

        # Try to get SL from open position for accurate R
        if pid in open_pnl:
            one_r = _one_r_from_position(open_pnl[pid])
        else:
            # Estimate from PnL: use a fixed assumption
            one_r = max(abs(pnl) / 5, 1.0) if abs(pnl) > 0.01 else 10.0

        r = pnl / one_r if one_r > 0 else 0
        total_r += r
        if pnl > 0.01: wins += 1
        elif pnl < -0.01: losses += 1
        else: breaks += 1

    trades = len(match_pids)
    avg_r = total_r / trades if trades > 0 else 0

    return {
        "error": None,
        "trades": trades,
        "pnl_r": round(total_r, 2),
        "avg_r": round(avg_r, 2),
        "wins": wins,
        "losses": losses,
    }
