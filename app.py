#!/usr/bin/env python3
"""
Momentum web app: runs a momentum script (NSE500_MOMENTUM_NOCAP.py or momentum_sectors.py) behind a small local website.

  pip install flask pandas numpy scipy yfinance requests
  python app.py            ->  open http://127.0.0.1:5000

Features
  * Refresh button + optional auto-refresh (every N minutes, optionally only during NSE market hours)
  * Incremental price updates (full download once a day, then only the last month is re-fetched)
  * Notes: a Notes column on every table, plus any extra columns you add; saved in notes.json, keyed by ticker
  * Settings (slots, sector cap, weighting, market filter) editable in the page
  * "Save official review" writes the same history files as the command-line script
Educational tool, not investment advice.
"""
import hmac
import json
import os
import struct
import zlib
import shutil
import threading
import time
import traceback
from argparse import Namespace
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory

import importlib
import sys

HERE = Path(__file__).resolve().parent
os.chdir(HERE)                                    # engine paths are relative to this folder
sys.path.insert(0, str(HERE))
IST = ZoneInfo("Asia/Kolkata")
NY = ZoneInfo("America/New_York")
CHOICE_FILE = HERE / "app_choice.json"

# key -> (label shown in the page, script file name without .py). Edit the names here if yours differ.
ENGINE_MODULES = {"nse": ("NSE 500 (India)", "NSE500_MOMENTUM_NOCAP"),
                  "us": ("S&P 500 (US)", "momentum_sectors")}


def _find_module(name):                           # match the file name ignoring upper/lower case
    for p in HERE.glob("*.py"):
        if p.stem.lower() == name.lower():
            return p.stem
    return None


ENGINES = {}
for _k, (_label, _mod) in ENGINE_MODULES.items():
    _found = _find_module(_mod)
    if not _found:
        print(f"[info] {_mod}.py not found next to app.py, so {_label} is unavailable")
        continue
    try:
        ENGINES[_k] = importlib.import_module(_found)
    except Exception as _e:                       # e.g. a package the script needs is not installed
        print(f"[warn] could not load {_found}.py ({_label}): {type(_e).__name__}: {_e}")
if not ENGINES:
    sys.exit("No momentum script found. Put NSE500_MOMENTUM_NOCAP.py and/or momentum_sectors.py next to app.py.")


def _default_settings(is_india):
    return {"top": 25, "max_sector": 1.0 if is_india else 0.30, "weighting": "equal", "market_filter": True,
            "buffer": 1.3, "max_weight": 0.08, "auto_minutes": 0, "market_hours_only": True,
            "capital": 0.0, "currency": "\u20b9", "fx": 1.0 if is_india else 0.0,
            "sector_alert": 0.40, "rank_alert": 10}


def _new_state():
    return {"running": False, "msg": "idle", "error": None, "started": None, "finished": None,
            "result": None, "price_time": None}


STATES = {k: _new_state() for k in ENGINES}
ACTIVE = None


def use_market(key):
    """Point the app at one market's script, folders and settings."""
    global ACTIVE, eng, IS_INDIA, BENCH_TICKERS, SETTINGS_FILE, NOTES_FILE, CACHE_FILE, DEFAULT_SETTINGS, STATE
    ACTIVE = key
    eng = ENGINES[key]
    IS_INDIA = hasattr(eng, "BENCH_TICKERS")      # the NSE script defines Indian benchmark tickers
    BENCH_TICKERS = getattr(eng, "BENCH_TICKERS", ["SPY"])
    SETTINGS_FILE = eng.DATA_DIR / "webapp_settings.json"
    NOTES_FILE = eng.DATA_DIR / "notes.json"
    CACHE_FILE = eng.DATA_DIR / "price_cache.pkl"
    DEFAULT_SETTINGS = _default_settings(IS_INDIA)
    STATE = STATES[key]
    eng.DATA_DIR.mkdir(parents=True, exist_ok=True)


def _initial_market():
    for a in sys.argv[1:]:                        # optional: `py app.py us`, `py app.py nse`, or a script file name
        a = a.lower().replace(".py", "")
        for k, (_l, m) in ENGINE_MODULES.items():
            if a in (k, m.lower()) and k in ENGINES:
                return k
    try:
        k = json.loads(CHOICE_FILE.read_text(encoding="utf-8")).get("market")
        if k in ENGINES:
            return k
    except Exception:
        pass
    return next(iter(ENGINES))


use_market(_initial_market())

app = Flask(__name__)

# ---------------------------------------------------------------- phone access: password + installable web app
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")      # set this before exposing the app beyond this computer
_PUBLIC = {"/manifest.webmanifest", "/sw.js", "/icon-192.png", "/icon-512.png"}   # harmless files the browser fetches without a login


@app.before_request
def _require_login():
    if not APP_PASSWORD or request.path in _PUBLIC:
        return None
    a = request.authorization                           # the browser shows a login box; any user name works
    if a and hmac.compare_digest((a.password or "").encode(), APP_PASSWORD.encode()):
        return None
    return Response("Login required", 401, {"WWW-Authenticate": 'Basic realm="Momentum"'})


def _icon_png(size):
    """Draw the app icon (blue square, white rising line) as a PNG, so no image files are needed."""
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    pts = [(.22, .68), (.40, .50), (.52, .60), (.78, .30)]
    d = np.full((size, size), 1e9)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        ax, ay, bx, by = x0 * size, y0 * size, x1 * size, y1 * size
        t = np.clip(((xx - ax) * (bx - ax) + (yy - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2), 0, 1)
        d = np.minimum(d, np.hypot(xx - (ax + t * (bx - ax)), yy - (ay + t * (by - ay))))
    a = np.clip(0.5 - (d - 0.045 * size), 0, 1)[..., None]          # soft edge, 1px wide
    bg, fg = np.array([47, 111, 222.0]), np.array([255.0, 255.0, 255.0])
    img = (bg * (1 - a) + fg * a).astype(np.uint8)
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(size))
    chunk = lambda k, b: struct.pack(">I", len(b)) + k + b + struct.pack(">I", zlib.crc32(k + b) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


_ICONS = {}


@app.get("/icon-<int:size>.png")
def icon(size):
    if size not in (192, 512):
        return Response("not found", 404)
    if size not in _ICONS:
        _ICONS[size] = _icon_png(size)
    return Response(_ICONS[size], mimetype="image/png", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/manifest.webmanifest")
def manifest():
    return jsonify({"name": "Momentum Dashboard", "short_name": "Momentum", "start_url": "/", "scope": "/",
                    "display": "standalone", "background_color": "#F5F7FA", "theme_color": "#2F6FDE",
                    "icons": [{"src": "/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
                              {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"}]})


@app.get("/sw.js")
def service_worker():                                   # a pass-through worker: lets the browser offer "Install app"; caches nothing
    return Response("self.addEventListener('install',e=>self.skipWaiting());\n"
                    "self.addEventListener('activate',e=>e.waitUntil(self.clients.claim()));\n"
                    "self.addEventListener('fetch',e=>{});\n", mimetype="text/javascript", headers={"Cache-Control": "no-cache"})

lock = threading.RLock()


# ---------------------------------------------------------------- storage helpers
def _read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def get_settings():
    s = dict(DEFAULT_SETTINGS)
    s.update(_read_json(SETTINGS_FILE, {}))
    return s


def make_cfg(s):
    """Build the config object the engine functions expect."""
    w = np.array([0.2, 0.4, 0.4])
    return Namespace(top=int(s["top"]), w1=w[0], w3=w[1], w6=w[2], buffer=float(s["buffer"]),
                     max_sector=float(s["max_sector"]), max_weight=max(float(s["max_weight"]), 1.0 / int(s["top"])),
                     weighting=s["weighting"], risk_adjust=True, winsor=3.0, min_price=20.0, min_dollar_vol=5e7 if IS_INDIA else 2e7,
                     market_filter=bool(s["market_filter"]), bear_exposure=0.5, review_days=14,
                     universe_file=None, no_refresh_universe=False)


# ---------------------------------------------------------------- prices (cached + incremental)
def _norm_index(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df[~df.index.duplicated(keep="last")]


def _recent(tickers):
    import yfinance as yf
    cs, vs = [], []
    for i in range(0, len(tickers), 250):
        raw = yf.download(tickers[i:i + 250], period="1mo", auto_adjust=True, progress=False, threads=True)
        c, v = raw["Close"], raw["Volume"]
        if isinstance(c, pd.Series):
            c, v = c.to_frame(name=tickers[i]), v.to_frame(name=tickers[i])
        cs.append(_norm_index(c))
        vs.append(_norm_index(v))
    return pd.concat(cs, axis=1), pd.concat(vs, axis=1)


def load_prices(uni, status):
    tickers = list(uni.index) + BENCH_TICKERS
    cache = None
    if CACHE_FILE.exists():
        try:
            cache = pd.read_pickle(CACHE_FILE)
        except Exception:
            cache = None
    fresh = cache and (time.time() - cache["full_at"] < 20 * 3600) and set(uni.index) <= set(cache["close"].columns)
    if fresh:
        status("Updating recent prices (incremental)...")
        nc, nv = _recent(tickers)
        close = nc.combine_first(cache["close"])
        vol = nv.combine_first(cache["vol"])
        full_at = cache["full_at"]
    else:
        status("Full price download (about 2 years, 500 stocks; takes a minute or two)...")
        c, v = eng.download_prices(tickers, 2)
        close, vol = _norm_index(c), _norm_index(v)
        full_at = time.time()
    close, vol = close.sort_index(), vol.reindex(index=close.index, columns=close.columns)
    pd.to_pickle({"close": close, "vol": vol, "full_at": full_at}, CACHE_FILE)

    bench = pd.Series(dtype=float)
    for b in BENCH_TICKERS:                   # same benchmark logic as the script
        s = close.pop(b) if b in close.columns else None
        if b in vol.columns:
            vol = vol.drop(columns=b)
        if s is not None and bench.dropna().shape[0] < 250 and s.dropna().shape[0] >= 250:
            bench = s
    return close, vol, bench


# ---------------------------------------------------------------- performance since each saved review
def _num(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def compute_perf(close, bench, asof, held_now, uni):
    """Hold each saved review's target weights from that day's close to the next review (or to now),
    and compare with the index. Everything is rebuilt from the saved snapshots plus cached prices."""
    snaps = []
    for p in sorted(eng.SNAP_DIR.glob("portfolio_*.csv")):
        d = p.stem.replace("portfolio_", "")[:10]
        try:
            df, cash = eng.read_snapshot(p)
            snaps.append((d, df, float(cash)))
        except Exception:
            continue
    snaps = [x for x in snaps if x[0] <= asof]
    if not snaps:
        return None, {}
    cf = close.ffill()
    bf = bench.reindex(cf.index).ffill() if len(bench.dropna()) else pd.Series(dtype=float)

    def at(frame, d):
        sub = frame.loc[:pd.Timestamp(d)]
        return sub.iloc[-1] if len(sub) else None

    def period(df, cash, d0, d1):
        p0, p1 = at(cf, d0), at(cf, d1)
        if p0 is None or p1 is None or pd.Timestamp(d0) < cf.index[0]:
            return None
        w = df["weight"].astype(float)
        r = p1.reindex(w.index) / p0.reindex(w.index) - 1
        ok = r.notna() & np.isfinite(r)
        contrib = w * r.where(ok, 0.0)
        strat, b = float(contrib.sum()), None          # cash is assumed to earn nothing
        if len(bf):
            b0, b1 = at(bf, d0), at(bf, d1)
            if b0 is not None and b1 is not None and pd.notna(b0) and pd.notna(b1) and b0:
                b = float(b1 / b0 - 1)
        return {"start": d0, "end": d1, "days": (pd.Timestamp(d1) - pd.Timestamp(d0)).days, "strat": strat, "bench": b,
                "excess": None if b is None else strat - b, "n": int(len(w)), "cash": cash, "missing": int((~ok).sum()),
                "hit": float((r[ok] > 0).mean()) if ok.any() else None,
                "_r": r, "_c": contrib, "_w": w, "_p0": p0, "_p1": p1}

    periods = []
    for i, (d, df, cash) in enumerate(snaps):
        d1 = snaps[i + 1][0] if i + 1 < len(snaps) else asof
        if d1 <= d:
            continue                                  # reviewed today: nothing to measure yet
        pr = period(df, cash, d, d1)
        if pr:
            pr["open"] = (i + 1 == len(snaps))
            periods.append(pr)

    cum, sv, bv = [], 100.0, 100.0
    if periods:
        cum.append({"date": periods[0]["start"], "strat": 100.0, "bench": 100.0})
        for pr in periods:
            sv *= 1 + pr["strat"]
            bv = None if (bv is None or pr["bench"] is None) else bv * (1 + pr["bench"])
            cum.append({"date": pr["end"], "strat": sv, "bench": bv})

    detail = periods[-1] if periods else None
    ret_by_t = {}
    if detail is not None and detail["open"]:
        ret_by_t = {t: _num(v) for t, v in detail["_r"].items()}
    out = {"bench_label": "Nifty 500 index" if IS_INDIA else "S&P 500 (SPY)", "last_review": snaps[-1][0],
           "reviewed_today": snaps[-1][0] == asof, "n_reviews": len(snaps),
           "history": [{k: v for k, v in pr.items() if not k.startswith("_")} for pr in periods], "cum": cum,
           "cum_strat": sv / 100 - 1 if periods else None,
           "cum_bench": None if (not periods or bv is None) else bv / 100 - 1, "detail": None}
    if detail is not None:
        names, secs = uni["security"].to_dict(), uni["sector"].to_dict()
        rows = []
        for t in detail["_w"].index:
            rows.append({"t": t, "name": names.get(t, t), "sector": secs.get(t, "Unknown"), "w": _num(detail["_w"][t]),
                         "p0": _num(detail["_p0"].get(t)), "p1": _num(detail["_p1"].get(t)), "ret": _num(detail["_r"].get(t)),
                         "contrib": _num(detail["_c"].get(t)), "held": t in held_now})
        rows.sort(key=lambda x: -(x["contrib"] if x["contrib"] is not None else -9))
        out["detail"] = {"start": detail["start"], "end": detail["end"], "open": detail["open"], "rows": rows}
    return out, ret_by_t


# ---------------------------------------------------------------- the analysis
def run_analysis(record=False):
    def status(m):
        with lock:
            STATE["msg"] = m

    s = get_settings()
    cfg = make_cfg(s)
    status("Loading stock universe...")
    uni = eng.get_universe(cfg)
    close, vol, bench = load_prices(uni, status)
    status("Scoring...")
    close, vol = eng.clean_data(close, vol)
    asof = close.index[-1].strftime("%Y-%m-%d")
    f = eng.score_universe(eng.compute_features(close, vol), cfg)
    label, prev, cash_old = eng.latest_snapshot(asof)
    prev_hold = prev.index if prev is not None else []
    w, risk_on = eng.choose_portfolio(f, uni["sector"], prev_hold, bench, cfg)
    cash_new = 1.0 - float(w.sum())

    port = w.to_frame().join(f[["rank", "erank", "r1", "r3", "r6", "vol", "score"]]).join(uni[["security", "sector"]])
    port.index.name = "ticker"
    port["status"] = np.where(port["erank"] > cfg.top, "extended", "core")
    port = port.sort_values("weight", ascending=False)
    changes = eng.compare(prev, w, uni, f, cfg, cash_old if prev is not None else 0.0, cash_new)
    due, last, next_due = eng.review_status(datetime.now(), cfg)

    saved = None
    if record:                                    # same files the CLI writes
        for kind in ("portfolio", "ranking"):
            p = eng.SNAP_DIR / f"{kind}_{asof}.csv"
            if p.exists():
                eng.ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy(p, eng.ARCHIVE_DIR / f"{kind}_{asof}_{datetime.now():%H%M%S}.csv")
        cash_row = pd.DataFrame({"weight": [cash_new], "security": ["Cash"], "sector": ["Cash"]},
                                index=pd.Index([eng.CASH], name="ticker"))
        out = pd.concat([port, cash_row])
        out.index.name = "ticker"
        eng.save_csv(out, eng.SNAP_DIR / f"portfolio_{asof}.csv")
        ranked = f.join(uni[["security", "sector"]])
        ranked.index.name = "ticker"
        eng.save_csv(ranked, eng.SNAP_DIR / f"ranking_{asof}.csv")
        eng.save_csv(changes, eng.CHANGES_DIR / f"changes_{asof}.csv")
        saved = asof
        due, last, next_due = eng.review_status(datetime.now() + timedelta(days=1), cfg)

    names, secs = uni["security"].to_dict(), uni["sector"].to_dict()
    held = set(w.index)
    capital, fx = float(s.get("capital") or 0), float(s.get("fx") or 0)

    prev_rank = pd.Series(dtype=float)            # ranks at the last official review, to show movers
    if label is not None:
        pf = eng.SNAP_DIR / f"ranking_{str(label)[:10]}.csv"
        if pf.exists():
            try:
                prev_rank = pd.read_csv(pf, index_col="ticker")["rank"]
                prev_rank = prev_rank[~prev_rank.index.duplicated()]
            except Exception:
                prev_rank = pd.Series(dtype=float)

    def rchg(t, rank):                            # positive = moved UP the ranking
        if t in prev_rank.index and pd.notna(prev_rank[t]) and pd.notna(rank):
            return float(prev_rank[t] - rank)
        return None

    perf, perf_ret = compute_perf(close, bench, asof, held, uni)
    holdings = []
    for t, r in port.iterrows():
        px = f.at[t, "price"]
        amt = r["weight"] * capital if capital > 0 else None
        sh = int(amt // (px * fx)) if (amt is not None and fx > 0 and px > 0) else None
        holdings.append({"t": t, "name": r["security"], "sector": r["sector"], "w": r["weight"], "rank": r["rank"],
                         "rchg": rchg(t, r["rank"]), "sret": perf_ret.get(t), "erank": r["erank"], "px": px, "amt": amt, "shares": sh,
                         "vsma": px / f.at[t, "ma200"] - 1, "r1": r["r1"], "r3": r["r3"], "r6": r["r6"],
                         "vol": r["vol"], "score": r["score"], "status": r["status"]})
    valid = f[f["valid"]]
    universe = [{"t": t, "name": names.get(t, t), "sector": secs.get(t, "Unknown"), "rank": r["rank"],
                 "rchg": rchg(t, r["rank"]),
                 "erank": r["erank"] if r["eligible"] else None, "px": r["price"], "r1": r["r1"], "r3": r["r3"],
                 "r6": r["r6"], "vsma": r["price"] / r["ma200"] - 1, "vol": r["vol"], "score": r["score"],
                 "eligible": bool(r["eligible"]), "held": t in held} for t, r in valid.iterrows()]
    watch = [u for u in sorted((u for u in universe if u["eligible"] and not u["held"]), key=lambda u: u["erank"])][:20]
    ch = changes[changes["action"].isin(["ADD", "REMOVE", "HOLD"])]
    chrows = []
    for t, r in ch.iterrows():
        px = f.at[t, "price"] if t in f.index else None
        amt = r["change"] * capital if capital > 0 else None
        sh = int(amt / (px * fx)) if (amt is not None and fx > 0 and px is not None and px > 0) else None
        chrows.append({"t": t, "name": r["security"], "sector": r["sector"], "action": r["action"], "old": r["old_weight"],
                       "new": r["new_weight"], "chg": r["change"], "px": px, "amt": amt, "tsh": sh, "reason": r["reason"]})

    sec = []
    if len(port):
        g = port.groupby("sector")["weight"].agg(n="count", w="sum").sort_values("w", ascending=False)
        sec = [{"sector": k, "n": int(v["n"]), "w": float(v["w"])} for k, v in g.iterrows()]
    thr = float(s.get("sector_alert") or 0.40)
    rthr = int(s.get("rank_alert") or 10)
    alerts = [x for x in sec if x["w"] >= thr]
    movers = sum(1 for h in holdings if h["rchg"] is not None and abs(h["rchg"]) >= rthr)
    cards = {"alerts": alerts, "sector_alert": thr, "movers": movers, "asof": asof, "risk_on": bool(risk_on), "holdings": len(port), "top": cfg.top, "cash": cash_new,
             "scored": int(f["valid"].sum()), "eligible": int(f["eligible"].sum()),
             "top_sector": sec[0] if sec else None, "compared_with": label,
             "review_due": bool(due), "next_due": next_due.strftime("%Y-%m-%d") if next_due is not None else None,
             "last_review": last.strftime("%Y-%m-%d") if last is not None else None, "saved": saved}
    if label is not None:
        cards["turnover"] = float(changes["change"].abs().sum() / 2)
    return {"cards": cards, "holdings": holdings, "universe": universe, "watch": watch, "changes": chrows,
            "sectors": sec, "perf": perf, "settings": s}


def _worker(record):
    try:
        res = eng._clean(run_analysis(record))
        with lock:
            STATE.update(result=res, error=None, msg="done", finished=time.time())
    except SystemExit as e:                       # engine calls sys.exit() when the universe cannot be fetched
        with lock:
            STATE.update(error=str(e) or "run aborted", msg="failed", finished=time.time())
    except Exception as e:
        traceback.print_exc()
        with lock:
            STATE.update(error=f"{type(e).__name__}: {e}", msg="failed", finished=time.time())
    finally:
        with lock:
            STATE["running"] = False


def start_run(record=False):
    with lock:
        if STATE["running"]:
            return False
        STATE.update(running=True, error=None, started=time.time(), msg="starting...")
    threading.Thread(target=_worker, args=(record,), daemon=True).start()
    return True


def market_tz():
    return IST if IS_INDIA else NY


def market_open(now=None):
    now = now or datetime.now(market_tz())
    lo, hi = ((9, 15), (15, 45)) if IS_INDIA else ((9, 30), (16, 15))   # NSE in IST; US in New York time
    return now.weekday() < 5 and lo <= (now.hour, now.minute) <= hi


def scheduler():
    while True:
        time.sleep(5)
        try:
            s = get_settings()
            m = int(s.get("auto_minutes") or 0)
            if m <= 0:
                continue
            with lock:
                busy, last = STATE["running"], (STATE["finished"] or STATE["started"] or 0)
            if busy or time.time() - last < m * 60:
                continue
            if s.get("market_hours_only") and not market_open():
                continue
            start_run(False)
        except Exception:
            traceback.print_exc()


# ---------------------------------------------------------------- notes (columns + values, keyed by ticker)
def get_notes():
    n = _read_json(NOTES_FILE, {})
    n.setdefault("columns", ["Notes"])
    n.setdefault("data", {})
    return n


# ---------------------------------------------------------------- routes
@app.get("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.get("/api/markets")
def api_markets():
    return jsonify({"markets": [{"key": k, "label": ENGINE_MODULES[k][0]} for k in ENGINES], "active": ACTIVE})


@app.post("/api/market")
def api_market():
    key = (request.get_json(silent=True) or {}).get("key")
    with lock:
        if key not in ENGINES:
            return jsonify({"ok": False, "error": "unknown market"})
        if STATE["running"]:
            return jsonify({"ok": False, "error": "A refresh is running. Wait for it to finish, then switch."})
        use_market(key)
        _write_json(CHOICE_FILE, {"market": key})
    return jsonify({"ok": True, "active": key})


@app.get("/api/status")
def api_status():
    with lock:
        return jsonify({k: STATE[k] for k in ("running", "msg", "error", "started", "finished")} |
                       {"has_data": STATE["result"] is not None, "market_open": market_open(),
                        "now": datetime.now(market_tz()).strftime("%a %H:%M ") + ("IST" if IS_INDIA else "New York"),
                        "market": ACTIVE})


@app.get("/api/data")
def api_data():
    with lock:
        return jsonify({"result": STATE["result"], "finished": STATE["finished"]})


@app.post("/api/refresh")
def api_refresh():
    body = request.get_json(silent=True) or {}
    return jsonify({"started": start_run(bool(body.get("record")))})


@app.get("/api/settings")
def api_get_settings():
    return jsonify(get_settings())


@app.post("/api/settings")
def api_set_settings():
    s = get_settings()
    body = request.get_json(silent=True) or {}
    for k in DEFAULT_SETTINGS:
        if k in body:
            s[k] = type(DEFAULT_SETTINGS[k])(body[k]) if not isinstance(DEFAULT_SETTINGS[k], bool) else bool(body[k])
    s["top"] = max(5, min(100, int(s["top"])))
    _write_json(SETTINGS_FILE, s)
    return jsonify(s)


@app.get("/api/notes")
def api_get_notes():
    return jsonify(get_notes())


@app.post("/api/notes")
def api_set_note():
    b = request.get_json(silent=True) or {}
    with lock:
        n = get_notes()
        t, c, v = str(b.get("t", "")), str(b.get("col", "")), str(b.get("value", ""))
        if t and c in n["columns"]:
            row = n["data"].setdefault(t, {})
            if v:
                row[c] = v
            else:
                row.pop(c, None)
            _write_json(NOTES_FILE, n)
    return jsonify({"ok": True})


@app.post("/api/columns")
def api_columns():
    b = request.get_json(silent=True) or {}
    name = str(b.get("name", "")).strip()[:40]
    with lock:
        n = get_notes()
        if b.get("action") == "add" and name and name not in n["columns"]:
            n["columns"].append(name)
        elif b.get("action") == "remove" and name in n["columns"] and len(n["columns"]) > 1:
            n["columns"].remove(name)
            for row in n["data"].values():
                row.pop(name, None)
        _write_json(NOTES_FILE, n)
    return jsonify(n)


if __name__ == "__main__":
    HOST, PORT = os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", "5000"))
    if HOST not in ("127.0.0.1", "localhost") and not APP_PASSWORD and os.environ.get("ALLOW_OPEN") != "1":
        sys.exit("Refusing to listen beyond this computer without a password. Set APP_PASSWORD first "
                 "(Windows: set APP_PASSWORD=yourpassword).")
    threading.Thread(target=scheduler, daemon=True).start()
    print(f"Markets available: {', '.join(ENGINE_MODULES[k][0] for k in ENGINES)}. Starting on: {ENGINE_MODULES[ACTIVE][0]}")
    print(f"Open http://127.0.0.1:{PORT}  (Ctrl+C to stop)" + (f"   Phone on same Wi-Fi: http://<this-PC-IP>:{PORT}" if HOST == "0.0.0.0" else ""))
    app.run(host=HOST, port=PORT, debug=False, threaded=True)


if __name__ != "__main__" and os.environ.get("MOMENTUM_SERVE") == "1":   # started by gunicorn (see Dockerfile)
    threading.Thread(target=scheduler, daemon=True).start()
