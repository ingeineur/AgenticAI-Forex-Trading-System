"""Manage strategy subprocesses — start, stop, logs, status."""
import os
import subprocess
import signal
import time
import json
import secrets
import string
from .mt5_pnl import get_pnl

CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'configs')
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')

_processes = {}  # id -> StrategyProcess


def _short_id():
    return ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(5))


def _log_path(instance_id):
    return os.path.join(LOG_DIR, f"{instance_id}.log")


class StrategyProcess:
    def __init__(self, instance_id, config):
        self.id = instance_id
        self.config = config
        if not self.config.get('short_id'):
            self.config['short_id'] = _short_id()
        self.proc = None
        self.status = "stopped"
        self.started_at = None
        self.last_signal = None
        self._log_buf = []

    def start(self):
        if self.proc is not None and self.proc.poll() is None:
            return False
        cmd = ["python", self.config["file"], self.config["symbol"],
               "--watch", "--trade", "--tf", self.config["tf"]]
        cmd += self.config.get("args", [])
        cmd += ["--instance-id", self.config['short_id']]
        log_p = _log_path(self.id)
        self._log_fh = open(log_p, "a", encoding="utf-8")
        self.proc = subprocess.Popen(
            cmd, cwd=os.path.dirname(os.path.dirname(__file__)),
            stdout=self._log_fh, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
        self.status = "running"
        self.started_at = time.time()
        return True

    def stop(self):
        if self.proc is None or self.proc.poll() is not None:
            self.status = "stopped"
            return True
        try:
            os.kill(self.proc.pid, signal.CTRL_BREAK_EVENT)
        except:
            self.proc.terminate()
        self.proc.wait(timeout=10)
        self.status = "stopped"
        if hasattr(self, '_log_fh'):
            self._log_fh.close()
        return True

    def restart(self):
        self.stop()
        time.sleep(1)
        return self.start()

    @property
    def running_time(self):
        if self.status == "running" and self.started_at:
            return int(time.time() - self.started_at)
        return 0

    def tail_log(self, n=50):
        log_dir = os.path.dirname(os.path.dirname(__file__))
        _LOG_MAP = {"sweep_mss_zone.py": "sweep_mss", "sweep_mss_zone_v2.py": "sweep_mss_v2",
                    "sweep_mss_zone_v3.py": "sweep_mss_v3", "sweep_mss_zone_v4.py": "sweep_mss_v4",
                    "smc_strategy.py": "smc"}
        log_prefix = _LOG_MAP.get(self.config.get('file', ''), '')
        symbol = self.config.get('symbol', '')
        tf = self.config.get('tf', '')
        strat_log = os.path.join(log_dir, f"{symbol}_{tf}_{log_prefix}.log") if log_prefix and tf else None
        log_p = _log_path(self.id)
        all_lines = []
        for fp in [strat_log, log_p]:
            if fp and os.path.exists(fp):
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        all_lines.extend(f.readlines())
                except: pass
        lines = [l.rstrip("\n\r") for l in all_lines[-n:]]
        return lines

    def info(self):
        try:
            pnl_data = get_pnl(self.config.get('short_id', ''))
        except:
            pnl_data = {"error": "MT5 error", "trades": 0, "pnl_r": 0, "avg_r": 0, "wins": 0, "losses": 0}
        bias = self._parse_bias()
        return {
            "id": self.id,
            "short_id": self.config.get('short_id'),
            "file": self.config.get("file"),
            "symbol": self.config.get("symbol"),
            "tf": self.config.get("tf"),
            "args": self.config.get("args", []),
            "model": self._model_name(),
            "volume": self._volume(),
            "swept": self._parse_setup_field("Swept"),
            "mss": self._parse_setup_field("MSS"),
            "status": self.status,
            "uptime": self.running_time,
            "last_signal": self.last_signal,
            "pnl": pnl_data,
            "bias": bias,
        }

    def _model_name(self):
        """Derive the ML model filename used by the strategy (honours --ml-path override)."""
        args = self.config.get("args", [])
        if "--ml-path" in args:
            i = args.index("--ml-path")
            if i + 1 < len(args):
                return os.path.basename(args[i + 1])
        f = self.config.get("file", "")
        sym = self.config.get("symbol", "")
        tf = self.config.get("tf", "")
        ver = {"sweep_mss_zone.py": "", "sweep_mss_zone_v2.py": "_v2",
               "sweep_mss_zone_v3.py": "_v3", "sweep_mss_zone_v4.py": "_v4"}.get(f, "")
        return f"sweep_ml_{sym}_{tf}{ver}.pkl"

    def _volume(self):
        args = self.config.get("args", [])
        if "--volume" in args:
            i = args.index("--volume")
            if i + 1 < len(args):
                return args[i + 1]
        return "0.01"

    def _log_files(self):
        """Strategy log file + instance log file (captures strategy stdout)."""
        log_dir = os.path.dirname(os.path.dirname(__file__))
        _LOG_MAP = {"sweep_mss_zone.py": "sweep_mss", "sweep_mss_zone_v2.py": "sweep_mss_v2",
                    "sweep_mss_zone_v3.py": "sweep_mss_v3", "sweep_mss_zone_v4.py": "sweep_mss_v4",
                    "smc_strategy.py": "smc"}
        log_prefix = _LOG_MAP.get(self.config.get('file', ''), '')
        symbol = self.config.get('symbol', '')
        tf = self.config.get('tf', '')
        strat_log = os.path.join(log_dir, f"{symbol}_{tf}_{log_prefix}.log") if log_prefix and tf else None
        log_p = _log_path(self.id)
        files = []
        if strat_log and os.path.exists(strat_log):
            files.append(strat_log)
        if os.path.exists(log_p):
            files.append(log_p)
        return files

    def _parse_setup_field(self, field):
        """Parse the latest 'Swept:' / 'MSS:' line from the setup signal logs."""
        for fp in self._log_files():
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    for line in reversed(f.readlines()):
                        if f"{field}:" not in line:
                            continue
                        val = line.split(f"{field}:", 1)[1].strip()
                        if val and val != "None":
                            return val
            except:
                pass
        return None

    def _parse_bias(self):
        """Parse latest SWING log line to extract bias values."""
        _files = self._log_files()

        _BIAS_MAP = {"1": "BULLISH", "-1": "BEARISH", "0": "NEUTRAL"}
        for fp in _files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    for line in reversed(f.readlines()):
                        if "SWING" not in line: continue
                        import re
                        # MSS format: trend=BEARISH tf_trend=BEARISH
                        m = re.search(r'trend=(\S+)\s+tf_trend=(\S+)', line)
                        if m:
                            fv = _BIAS_MAP.get(m.group(1), m.group(1))
                            tv = _BIAS_MAP.get(m.group(2), m.group(2))
                            return {"final": fv, "tf": tv}
                        # SMC format: bias=1 TF-bias=-1
                        m = re.search(r'bias=(\S+)\s+TF-bias=(\S+)', line)
                        if m:
                            fv = _BIAS_MAP.get(m.group(1), m.group(1))
                            tv = _BIAS_MAP.get(m.group(2), m.group(2))
                            return {"final": fv, "tf": tv}
            except:
                pass
        return None


def load_configs():
    """Load all saved strategy configs from disk."""
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)
    configs = {}
    for fname in os.listdir(CONFIG_DIR):
        if fname.endswith(".json"):
            with open(os.path.join(CONFIG_DIR, fname), "r") as f:
                cfg = json.load(f)
                iid = cfg.get("id", fname.replace(".json", ""))
                configs[iid] = cfg
                if iid not in _processes:
                    _processes[iid] = StrategyProcess(iid, cfg)
    return configs


def save_config(instance_id, config):
    path = os.path.join(CONFIG_DIR, f"{instance_id}.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def get_process(instance_id):
    return _processes.get(instance_id)


def list_processes():
    return [p.info() for p in _processes.values()]


def create_process(instance_id, config):
    if instance_id in _processes:
        return None
    proc = StrategyProcess(instance_id, config)
    _processes[instance_id] = proc
    save_config(instance_id, proc.config)  # save AFTER short_id is generated
    return proc


def delete_process(instance_id):
    proc = _processes.pop(instance_id, None)
    if proc:
        proc.stop()
    config_path = os.path.join(CONFIG_DIR, f"{instance_id}.json")
    if os.path.exists(config_path):
        os.remove(config_path)
