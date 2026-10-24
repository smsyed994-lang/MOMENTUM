#!/usr/bin/env python3
"""
Runs the momentum analysis for each market and writes data/<market>.json for the phone page.
This is what GitHub Actions runs (see .github/workflows/refresh.yml). You can also run it on your PC:

    py export_results.py              # all markets
    MARKET=nse py export_results.py   # one market  (Windows: set MARKET=nse)
    RECORD=1 py export_results.py     # also save an OFFICIAL review (same as the button)

Settings come from config.json, e.g. {"nse": {"top": 25, "max_sector": 1.0}, "us": {"max_sector": 0.3}}.
Your capital is NOT used here: the phone page works out rupee/dollar amounts itself, so it never leaves your phone.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import app                                        # reuses the exact analysis the web app runs

HERE = Path(__file__).resolve().parent


def main():
    cfg_path = HERE / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    want = (os.environ.get("MARKET") or "all").lower()
    record = os.environ.get("RECORD") in ("1", "true", "True")
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    ok, failed = [], []
    for key in app.ENGINES:
        if want not in ("all", key):
            continue
        app.use_market(key)
        base = app._default_settings(app.IS_INDIA)
        base.update(cfg.get(key, {}))
        base["capital"] = 0.0                     # sizing is done on the phone
        app.get_settings = lambda base=base: dict(base)
        print(f"== {app.ENGINE_MODULES[key][0]} (record={record})", flush=True)
        try:
            res = app.eng._clean(app.run_analysis(record=record))
        except SystemExit as e:                   # the script exits when it cannot fetch the stock list
            failed.append((key, str(e) or "run aborted"))
            continue
        except Exception as e:
            failed.append((key, f"{type(e).__name__}: {e}"))
            continue
        payload = {"market": key, "label": app.ENGINE_MODULES[key][0], "generated_at": time.time(),
                   "generated_iso": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "result": res}
        tmp = out / f"{key}.json.tmp"
        tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, out / f"{key}.json")
        ok.append(key)
        print(f"wrote data/{key}.json", flush=True)
    for key, why in failed:
        print(f"[FAILED] {key}: {why}", file=sys.stderr)
    if failed and not ok:
        sys.exit(1)                               # makes the GitHub run show red so the phone can tell you


if __name__ == "__main__":
    main()
