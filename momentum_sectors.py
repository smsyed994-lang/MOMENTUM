#!/usr/bin/env python3
"""
Biweekly multi-horizon momentum portfolio (1-month, 3-month, 6-month returns).

HOW THE SYSTEM WORKS
  1. Universe : S&P 500 constituents (or your own list via --universe-file).
  2. Data     : adjusted daily prices + volume from Yahoo Finance, quality-checked.
  3. Signal   : for every stock, compute the 1m (21d), 3m (63d) and 6m (126d)
                returns. Each is converted to a cross-sectional z-score (how
                strong is this stock versus all others?), capped at +/-3 so one
                outlier can't dominate, then blended:
                    score = 0.2*z(1m) + 0.4*z(3m) + 0.4*z(6m)      (adjustable)
                By default returns are first divided by volatility, so a smooth
                +30% beats a wild +30%.
  4. Filters  : a stock is only eligible if it is liquid, trades ABOVE its
                200-day average, and has a POSITIVE 6-month return.
  5. Portfolio: pick the top N eligible stocks with a cap on names per sector.
                Current holdings get priority while they stay within a rank
                "buffer" (reduces needless trading). EVERY SLOT HAS EQUAL WEIGHT
                (1/N each; --weighting invvol switches to inverse-volatility).
                Any slot that cannot be filled by an eligible stock stays in CASH.
  6. Market   : optional filter - if SPY is below its 200-day average, total
                stock exposure is scaled down (default: to 50%).
  7. Cadence  : a full review happens every 14 days. Run the script weekly (or
                daily); it does nothing until a review is due, unless --force.
  8. History  : every review is saved (holdings, full ranking, and a change
                report with the reason for each add/remove).
  9. Dashboard: each review (and each backtest) also writes an interactive HTML
                dashboard. It includes a Modern Portfolio Theory comparison:
                the same stocks equal-weighted versus min-variance and
                max-Sharpe optimised, the efficient frontier, and SPY.

FILES WRITTEN (momentum_data_biweekly/)
    universe.csv                         cached constituent list
    snapshots/portfolio_YYYY-MM-DD.csv   target holdings + weights (incl. CASH row)
    snapshots/ranking_YYYY-MM-DD.csv     every stock's features, score and rank
    snapshots/archive/                   copies of anything a same-day re-run replaced
    changes/changes_YYYY-MM-DD.csv       ADD / REMOVE / HOLD / CASH vs previous review
    reports/report_YYYY-MM-DD.html       interactive dashboard of an OFFICIAL review (open in a browser)
    reports/interim_YYYY-MM-DD.html      dashboard of an interim view (between reviews)
    reports/backtest_report.html         interactive backtest dashboard

THIS IS A SINGLE, SELF-CONTAINED FILE: no other .py files are needed. Name it anything you like.

USAGE
    pip install pandas numpy scipy yfinance requests lxml
    python momentum_biweekly.py                  # ALWAYS prints the full output + writes the dashboard.
                                                 # Due (14 days) -> official review, saved to history.
                                                 # Not due       -> "interim view", history untouched.
    python momentum_biweekly.py --open           # ... and open the dashboard in your browser
    python momentum_biweekly.py --force          # record an official review right now, even if not due
    python momentum_biweekly.py --health-check   # also print the holdings risk check
    python momentum_biweekly.py --backtest --open  # historical test vs SPY and MPT
    python momentum_biweekly.py --top 20 --weights 0.1 0.4 0.5 --no-market-filter

Educational tool, not investment advice. Backtests use today's index members
(survivorship bias), so real results will be worse than the backtest.
"""

# ----------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------
import argparse                                   # reads command-line flags such as --top 25
import io                                         # in-memory text buffer, used to hand HTML to pandas
import json                                       # turns Python data into JSON text for the dashboard
import math                                       # isfinite() to catch NaN / infinity when writing the dashboard
import os                                         # os.replace() lets us swap files atomically
import shutil                                     # shutil.copy() copies files (for the archive)
import webbrowser                                 # opens the finished dashboard in your default browser
import sys                                        # sys.exit() stops the program, sys.stderr is the warning stream
from datetime import datetime, timedelta          # dates for the 14-day review schedule
from pathlib import Path                          # clean, cross-platform file paths

import numpy as np                                # fast math on arrays (sqrt, floor, nan handling)
import pandas as pd                               # tables (DataFrames) for prices, scores and holdings

# ----------------------------------------------------------------------------
# CONSTANTS (the fixed "shape" of the system)
# ----------------------------------------------------------------------------
DATA_DIR = Path("momentum_data_biweekly")         # one folder holding everything this system saves
SNAP_DIR = DATA_DIR / "snapshots"                 # sub-folder for portfolio + ranking snapshots
ARCHIVE_DIR = SNAP_DIR / "archive"                # sub-folder for copies of overwritten same-day snapshots
CHANGES_DIR = DATA_DIR / "changes"                # sub-folder for add/remove change reports
UNIVERSE_FILE = DATA_DIR / "universe.csv"         # cached list of S&P 500 tickers
CASH = "CASH"                                     # special row name that stores the cash weight

HORIZONS = {"r1": 21, "r3": 63, "r6": 126}        # return windows in trading days: ~1, ~3, ~6 months
MA_DAYS = 200                                     # trend filter: 200-day moving average
VOL_DAYS = 63                                     # volatility measured over the last ~3 months
MIN_HISTORY = MA_DAYS + 1                         # minimum price rows a stock needs to be scored
STEP = 10                                         # backtest: rebalance every 10 trading days (~2 weeks)
REPORT_DIR = DATA_DIR / "reports"                 # where the HTML dashboards are written
LABELS = {"momentum": "Momentum (equal weight)",  # display names for the backtest strategies
          "mpt_sharpe": "Momentum + max-Sharpe (MPT)",
          "mpt_minvar": "Momentum + min-variance (MPT)",
          "universe": "S&P 500 stocks, equal weight",
          "SPY": "SPY (S&P 500 index)"}


# ----------------------------------------------------------------------------
# SMALL HELPERS
# ----------------------------------------------------------------------------
def warn(msg):                                    # print a warning without mixing into normal output
    print(f"[warn] {msg}", file=sys.stderr)       # send the text to the error stream


def save_csv(df, path):                           # write a CSV safely (never leaves a half-written file)
    path = Path(path)                             # make sure we have a Path object
    path.parent.mkdir(parents=True, exist_ok=True)  # create the folder if it does not exist yet
    tmp = path.with_suffix(".tmp")                # write to a temporary file first
    df.to_csv(tmp)                                # dump the table (index included) to the temp file
    os.replace(tmp, path)                         # swap it into place in one atomic step


# ----------------------------------------------------------------------------
# DATA: UNIVERSE AND PRICES
# ----------------------------------------------------------------------------
def get_universe(cfg):                            # returns a table indexed by ticker with security + sector
    DATA_DIR.mkdir(parents=True, exist_ok=True)   # make sure the data folder exists
    if cfg.universe_file:                         # the user supplied their own list of tickers
        uni = pd.read_csv(cfg.universe_file)      # read the CSV
        uni.columns = [c.strip().lower() for c in uni.columns]  # normalise column names
        if "ticker" not in uni.columns:           # we need at least a ticker column
            sys.exit("universe file needs a 'ticker' column")  # stop with a clear message
        uni["ticker"] = (uni["ticker"].astype(str).str.strip().str.upper()
                         .str.replace(".", "-", regex=False))  # Yahoo writes BRK.B as BRK-B
        uni = uni.drop_duplicates("ticker").set_index("ticker")  # one row per ticker, ticker as index
        if "security" not in uni.columns:         # name column is optional
            uni["security"] = uni.index.to_series()  # fall back to the ticker itself
        if "sector" not in uni.columns:           # sector column is optional
            uni["sector"] = "Unknown"             # everything in one bucket if not provided
        return uni[["security", "sector"]]        # keep just the two columns we use
    if not cfg.no_refresh_universe:               # normal path: fetch today's S&P 500 list
        try:
            import requests                       # imported here so it is only needed when fetching
            r = requests.get(                     # download the Wikipedia constituents page
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                headers={"User-Agent": "Mozilla/5.0 momentum-model"},  # Wikipedia rejects blank user agents
                timeout=30,                       # give up after 30 seconds
            )
            r.raise_for_status()                  # raise an error on HTTP failures
            t = pd.read_html(io.StringIO(r.text))[0]  # first table on the page is the constituent list
            uni = pd.DataFrame({                  # build our own clean table
                "ticker": t["Symbol"].str.replace(".", "-", regex=False),  # Yahoo-style tickers
                "security": t["Security"],        # company name
                "sector": t["GICS Sector"],       # sector used for the sector cap
            }).set_index("ticker")                # index by ticker
            save_csv(uni, UNIVERSE_FILE)          # cache it for offline use next time
            return uni                            # done
        except Exception as e:                    # any failure: fall back to the cache
            warn(f"could not fetch constituents ({e}); using cache")  # tell the user
    if UNIVERSE_FILE.exists():                    # cached copy available?
        return pd.read_csv(UNIVERSE_FILE, index_col="ticker")  # load it
    sys.exit("No universe available: fetch failed and no cached universe.csv")  # nothing to work with


def _fetch(tickers, years, threads):              # one yfinance call returning (close, volume) tables
    import yfinance as yf                         # imported here so the file loads without it installed
    raw = yf.download(list(tickers), period=f"{years}y", auto_adjust=True,
                      progress=False, threads=threads)  # adjusted for splits and dividends
    close, vol = raw["Close"], raw["Volume"]      # split the two price/volume blocks
    if isinstance(close, pd.Series):              # a single ticker can come back as a Series
        name = list(tickers)[0]                   # the only ticker requested
        close, vol = close.to_frame(name=name), vol.to_frame(name=name)  # convert to one-column tables
    keep = close.dropna(axis=1, how="all").columns  # tickers that actually returned data
    return close[keep], vol.reindex(columns=keep)  # drop empty tickers from both tables


def download_prices(tickers, years):              # robust download with retries for failures
    tickers = list(dict.fromkeys(tickers))        # remove duplicates, keep order
    close, vol = _fetch(tickers, years, threads=True)  # fast threaded bulk download first
    for attempt in range(3):                      # then up to 3 retry rounds for anything missing
        missing = [t for t in tickers if t not in close.columns]  # tickers we did not receive
        if not missing:                           # nothing missing: we are done
            break
        print(f"[info] retrying {len(missing)} missing ticker(s) one at a time "
              f"(attempt {attempt + 1}): {', '.join(missing[:10])}")  # progress message
        c2, v2 = _fetch(missing, years, threads=False)  # sequential download avoids cache-lock errors
        if not c2.empty:                          # if the retry returned anything
            close = close.join(c2, how="outer")   # add the new price columns
            vol = vol.join(v2, how="outer")       # add the new volume columns
    still = [t for t in tickers if t not in close.columns]  # tickers that never arrived
    if still:                                     # report them so nothing disappears silently
        warn(f"no price data for {len(still)} ticker(s), excluded: {', '.join(still)}")
    return close.sort_index(), vol.sort_index()   # both tables in date order


def clean_data(close, vol):                       # quality gate applied before any scoring
    close = close.sort_index().dropna(how="all")  # drop dates where no stock has a price
    close = close.where(close > 0)                # zero/negative prices are bad data -> NaN
    vol = vol.reindex(index=close.index, columns=close.columns)  # keep volume aligned to prices
    latest = close.index.max()                    # the most recent trading date in the data
    last_valid = pd.to_datetime(close.apply(lambda s: s.last_valid_index()))  # last date each stock had a price
    stale = last_valid.isna() | (last_valid < latest - pd.Timedelta(days=5))  # no recent price = delisted/halted
    short = close.notna().sum() < MIN_HISTORY     # too little history (recent IPO) to score fairly
    drop = stale | short                          # combine both reasons
    if drop.any():                                # tell the user what was excluded
        warn(f"dropped {int(drop.sum())} ticker(s) for stale/short data: "
             f"{', '.join(list(close.columns[drop.values])[:12])}")
    close = close.loc[:, ~drop].ffill(limit=3)    # keep good tickers; fill gaps of up to 3 days (holidays/halts)
    vol = vol.loc[:, close.columns]               # drop the same tickers from the volume table
    jumps = close.pct_change().abs().iloc[-126:].max()  # biggest one-day move per stock in ~6 months
    big = jumps[jumps > 0.40]                     # moves over 40% are suspicious (data glitch or real event)
    if len(big):                                  # flag them for a human to glance at
        warn(f"{len(big)} ticker(s) had a >40% one-day move in 6 months "
             f"(check data): {', '.join(big.index[:10])}")
    return close, vol                             # cleaned tables


# ----------------------------------------------------------------------------
# SIGNAL: FEATURES, Z-SCORES, COMPOSITE SCORE
# ----------------------------------------------------------------------------
def compute_features(close, vol):                 # raw numbers for every stock as of the last row
    if len(close) < MIN_HISTORY:                  # not enough dates to compute a 200-day average
        raise ValueError(f"need at least {MIN_HISTORY} trading days of prices")
    last = close.iloc[-1]                         # latest price of every stock
    feats = pd.DataFrame(index=close.columns)     # result table, one row per stock
    feats["price"] = last                         # store the latest price
    for name, days in HORIZONS.items():           # loop over r1, r3, r6
        feats[name] = last / close.iloc[-1 - days] - 1  # return over that many trading days
    daily = close.pct_change()                    # daily percentage returns
    feats["vol"] = daily.iloc[-VOL_DAYS:].std() * np.sqrt(252)  # annualised volatility (3 months)
    feats["ma200"] = close.iloc[-MA_DAYS:].mean() # 200-day simple moving average
    feats["dollar_vol"] = (close * vol).iloc[-20:].median()  # typical daily traded value (liquidity)
    return feats                                  # table of features


def zscore(s, clip):                              # how many standard deviations from the average?
    s = s.replace([np.inf, -np.inf], np.nan)      # infinite values (from zero division) become missing
    sd = s.std()                                  # spread of the values
    if not sd or np.isnan(sd) or sd == 0:         # no spread = no information
        return s * 0.0                            # everyone scores zero
    return ((s - s.mean()) / sd).clip(-clip, clip)  # standardise, then cap extremes at +/- clip


def score_universe(feats, cfg):                   # turns features into a ranked, filtered table
    f = feats.copy()                              # work on a copy, keep the input untouched
    need = ["r1", "r3", "r6", "vol", "ma200", "price"]  # numbers every scored stock must have
    f["valid"] = (f[need].notna().all(axis=1)     # no missing inputs
                  & (f["dollar_vol"] >= cfg.min_dollar_vol)  # liquid enough to trade
                  & (f["vol"] > 0))               # volatility must be positive (avoids divide-by-zero)
    base = f[f["valid"]]                          # only valid stocks take part in the z-scores
    z = {}                                        # holds the z-score of each horizon
    for name, days in HORIZONS.items():           # r1, r3, r6
        x = base[name]                            # that horizon's raw return
        if cfg.risk_adjust:                       # optional: return per unit of risk
            x = x / (base["vol"] * np.sqrt(days / 252))  # scale vol to the horizon's length
        z[name] = zscore(x, cfg.winsor)           # standardise and cap
    f["z1"], f["z3"], f["z6"] = z["r1"], z["r3"], z["r6"]  # store them (invalid stocks stay NaN)
    f["score"] = cfg.w1 * f["z1"] + cfg.w3 * f["z3"] + cfg.w6 * f["z6"]  # weighted blend = composite momentum
    f["above_ma"] = f["price"] > f["ma200"]       # is the stock in an uptrend?
    f["eligible"] = (f["valid"] & f["above_ma"]   # valid AND above its 200-day average
                     & (f["r6"] > 0)              # AND up over the last 6 months
                     & f["score"].notna())        # AND actually has a score
    f["rank"] = f["score"].rank(ascending=False, method="first")  # 1 = strongest momentum among all valid stocks
    f["erank"] = f["score"].where(f["eligible"]).rank(ascending=False, method="first")  # rank among ELIGIBLE stocks only
    return f.sort_values("score", ascending=False, na_position="last")  # best first, invalid last


# ----------------------------------------------------------------------------
# PORTFOLIO CONSTRUCTION
# ----------------------------------------------------------------------------
def select_picks(f, sectors, prev_holdings, cfg):  # decide which tickers to hold
    prev = set(prev_holdings)                     # current holdings as a set for fast lookup
    elig = f[f["eligible"]]                       # candidates, already sorted best-first
    sec = sectors.reindex(elig.index).fillna("Unknown")  # each candidate's sector
    cap = max(1, int(np.floor(cfg.top * cfg.max_sector)))  # max number of names allowed per sector
    picks, count = [], {}                         # chosen tickers, and how many we hold per sector

    def try_add(t):                               # add a ticker if there is room and the sector allows it
        s = sec[t]                                # that ticker's sector
        if len(picks) < cfg.top and t not in picks and count.get(s, 0) < cap:  # room + not duplicate + sector ok
            picks.append(t)                       # accept it
            count[s] = count.get(s, 0) + 1        # update that sector's tally

    zone = int(round(cfg.top * cfg.buffer))       # incumbents are kept while ranked inside this many names
    for t in elig.index[:zone]:                   # walk the top 'zone' candidates in rank order
        if t in prev:                             # only the ones we already own
            try_add(t)                            # keep them first (this is the turnover buffer)
    for t in elig.index:                          # then fill remaining slots from the best candidates
        try_add(t)                                # stops itself when top N is reached
    return picks                                  # fewer than N picks means the rest stays in cash


def cap_weights(w, total, cap):                   # scale weights to sum to `total`, none above `cap`
    w = w / w.sum() * total                       # proportional scaling to the target total
    if not cap:                                   # no cap requested
        return w
    if cap * len(w) <= total + 1e-12:             # cap too tight to ever reach the total
        return pd.Series(float(cap), index=w.index)  # everyone gets the cap (rest stays cash)
    for _ in range(50):                           # iteratively push excess to uncapped names
        over = w > cap + 1e-12                    # names above the cap
        if not over.any():                        # nobody over: finished
            break
        excess = (w[over] - cap).sum()            # total weight that must be moved
        w[over] = cap                             # clamp the big ones
        under = w < cap - 1e-12                   # names with room to take more
        if not under.any():                       # nowhere to put it: stop
            break
        w[under] += excess * w[under] / w[under].sum()  # share the excess pro-rata
    return w


def build_weights(f, picks, cfg):                 # turn the chosen tickers into portfolio weights
    if not picks:                                 # nothing qualified: 100% cash
        return pd.Series(dtype=float, name="weight")
    invested = len(picks) / cfg.top               # every slot is worth 1/N; unfilled slots stay cash
    if cfg.weighting == "invvol":                 # calmer stocks get bigger weights
        raw = 1.0 / f.loc[picks, "vol"]           # inverse volatility
    else:                                         # plain equal weights
        raw = pd.Series(1.0, index=picks)
    return cap_weights(raw, invested, cfg.max_weight).rename("weight")  # apply the per-stock cap


def market_risk_on(bench):                        # True if the market is in an uptrend
    b = bench.dropna()                            # ignore missing benchmark days
    if len(b) < MA_DAYS:                          # not enough data to judge
        return True                               # don't force defensive mode on missing data
    return bool(b.iloc[-1] > b.iloc[-MA_DAYS:].mean())  # SPY above its 200-day average?


def choose_portfolio(f, sectors, prev_holdings, bench, cfg):  # picks -> weights -> market filter
    picks = select_picks(f, sectors, prev_holdings, cfg)  # which stocks
    w = build_weights(f, picks, cfg)              # how much of each
    risk_on = market_risk_on(bench)               # market regime
    if cfg.market_filter and not risk_on:         # defensive mode
        w = w * cfg.bear_exposure                 # scale every weight down; remainder becomes cash
    return w[w > 1e-9], risk_on                   # drop zero weights and return the regime flag


# ----------------------------------------------------------------------------
# SNAPSHOTS, SCHEDULE AND COMPARISON
# ----------------------------------------------------------------------------
def review_status(today, cfg):                    # is a biweekly review due?
    files = sorted(SNAP_DIR.glob("portfolio_*.csv"))  # all saved reviews, oldest to newest
    if not files:                                 # never reviewed before
        return True, None, None                   # first review is always due
    last = datetime.strptime(files[-1].stem.replace("portfolio_", ""), "%Y-%m-%d")  # date of newest review
    next_due = last + timedelta(days=cfg.review_days)  # when the next one is scheduled
    due = (today - last).days >= cfg.review_days - 1   # allow 1 day of slack (Friday-to-Friday = 14)
    return due, last, next_due                    # answer plus dates for the status message


def read_snapshot(path):                          # load a saved portfolio and separate out the cash
    df = pd.read_csv(path, index_col="ticker")    # read the file
    if CASH in df.index:                          # cash row present
        cash = float(df.loc[CASH, "weight"])      # read it
    else:                                         # otherwise infer it
        cash = max(0.0, 1.0 - float(df["weight"].sum()))
    return df.drop(index=CASH, errors="ignore"), cash  # stocks only, plus cash separately


def latest_snapshot(asof, inclusive=False):       # find the review to compare against
    files = sorted(SNAP_DIR.glob("portfolio_*.csv"))  # all saved reviews
    stamp = lambda p: p.stem.replace("portfolio_", "")  # file name -> date string
    earlier = [p for p in files if (stamp(p) <= asof if inclusive else stamp(p) < asof)]  # reviews before (or, if inclusive, up to) this date
    if earlier:                                   # normal case
        p, label = earlier[-1], stamp(earlier[-1])  # the most recent earlier review
    else:                                         # no earlier review
        same = [p for p in files if stamp(p) == asof]  # maybe a same-day run exists
        if not same:                              # truly the first review
            return None, None, 0.0
        p, label = same[-1], f"{asof} (earlier run, same day)"  # compare to it, labelled clearly
    df, cash = read_snapshot(p)                   # load it
    return label, df, cash                        # label for display, holdings, cash


def exit_reason(t, f, cfg):                       # explain why a held stock is leaving
    if t not in f.index or not bool(f.at[t, "valid"]):  # no usable data
        return "no valid price data or too illiquid"
    r = f.loc[t]                                  # that stock's row
    if not r["above_ma"]:                         # trend broke
        return "closed below its 200-day average"
    if r["r6"] <= 0:                              # long-term momentum gone
        return "6-month return turned negative"
    if r["erank"] > round(cfg.top * cfg.buffer):  # fell too far down the ranking of eligible stocks
        return f"momentum rank fell to {int(r['erank'])} among eligible stocks (outside buffer)"
    return "replaced by stronger candidates or sector cap"  # still decent, but pushed out


def compare(prev, new, names, f, cfg, cash_old, cash_new):  # build the add/remove/hold report
    old_w = prev["weight"] if prev is not None else pd.Series(dtype=float)  # previous weights
    df = pd.DataFrame({"old_weight": old_w, "new_weight": new}).fillna(0.0)  # side by side, 0 = not held
    df["change"] = df["new_weight"] - df["old_weight"]  # weight change
    df["action"] = np.where(df["old_weight"].eq(0), "ADD",       # not held before -> ADD
                   np.where(df["new_weight"].eq(0), "REMOVE", "HOLD"))  # not held now -> REMOVE
    df["security"] = names["security"].reindex(df.index)  # company name
    df["sector"] = names["sector"].reindex(df.index)      # sector

    def reason(t, action):                        # short explanation for each row
        if action == "ADD":                       # new position
            ok = t in f.index and pd.notna(f.at[t, "erank"])  # do we have an eligible rank for it?
            return f"entered at eligible rank {int(f.at[t, 'erank'])}" if ok else "new position"
        if action == "REMOVE":                    # exiting position
            return exit_reason(t, f, cfg)
        return ""                                 # holds need no explanation

    df["reason"] = [reason(t, a) for t, a in zip(df.index, df["action"])]  # fill the reason column
    order = {"ADD": 0, "REMOVE": 1, "HOLD": 2}    # display order
    df = df.sort_values(by=["action", "change"],  # ADDs first, then REMOVEs, then HOLDs
                        key=lambda s: s.map(order) if s.name == "action" else -s.abs())  # biggest changes first
    df.index.name = "ticker"                      # name the index for the CSV
    cash = pd.DataFrame({"old_weight": [cash_old], "new_weight": [cash_new],
                         "change": [cash_new - cash_old], "action": ["CASH"],
                         "security": ["Cash"], "sector": ["Cash"], "reason": [""]},
                        index=pd.Index([CASH], name="ticker"))  # cash as its own row
    return pd.concat([df, cash])                  # stocks followed by the cash row


# ----------------------------------------------------------------------------
# REPORTING
# ----------------------------------------------------------------------------
def print_portfolio(port, cfg):                   # human-readable target portfolio
    show = port[["security", "sector", "weight", "rank", "r1", "r3", "r6", "vol", "status"]].copy()  # columns to show
    show["security"] = show["security"].astype(str).str[:24]  # trim long names
    show["rank"] = show["rank"].astype(int)       # whole-number ranks
    for c in ("weight", "r1", "r3", "r6", "vol"): # format the percentage columns
        show[c] = show[c].map("{:.1%}".format)    # e.g. 0.0412 -> 4.1%
    print(f"\nTarget portfolio ({len(port)} stocks; r1/r3/r6 = 1/3/6-month returns; rank = momentum rank "
          f"among all stocks; 'extended' = outside the top {cfg.top} eligible: kept via the buffer, "
          f"or added because a sector cap blocked a higher-ranked stock)")
    print(show.to_string() if len(show) else "(no stocks - 100% cash)")  # the table itself


def print_sector_summary(port, cash):             # how the portfolio is spread across sectors
    if len(port):                                 # only if we hold something
        g = port.groupby("sector")["weight"].agg(names="count", weight="sum")  # count and total weight per sector
        g = g.sort_values("weight", ascending=False)  # biggest sector first
        g["weight"] = g["weight"].map("{:.1%}".format)  # format as percent
        print("\nSector allocation")
        print(g.to_string())
    print(f"Cash: {cash:.1%}")                    # always show cash


def print_changes(ch, label):                     # what to buy and sell versus last review
    stocks = ch[ch["action"] != "CASH"]           # stock rows only
    cash = ch.loc[CASH]                           # the cash row
    print(f"\nCompared with previous review: {label or 'none (first review)'}")
    for action in ("ADD", "REMOVE"):              # the two lists that need action
        sub = stocks[stocks["action"] == action]  # rows for this action
        print(f"\n{action} ({len(sub)})")
        for t, r in sub.iterrows():               # one line per stock
            w = r["new_weight"] if action == "ADD" else r["old_weight"]  # show the relevant weight
            print(f"  {t:<7} {str(r['security'])[:24]:<24} {str(r['sector'])[:20]:<20} "
                  f"{w:6.2%}  {r['reason']}")
    hold = stocks[stocks["action"] == "HOLD"]     # positions kept
    print(f"\nHOLD ({len(hold)}) - largest weight changes")
    for t, r in hold.head(8).iterrows():          # only the 8 biggest resizes
        print(f"  {t:<7} {r['old_weight']:6.2%} -> {r['new_weight']:6.2%} ({r['change']:+.2%})")
    if label is None:                             # first review: nothing to compare with
        print(f"\nCASH: {cash['new_weight']:.1%}")
        print("Initial build: all positions are new.")
    else:                                         # later reviews: show turnover
        print(f"\nCASH: {cash['old_weight']:.1%} -> {cash['new_weight']:.1%}")
        turnover = (stocks["change"].abs().sum() + abs(cash["change"])) / 2  # one-way traded fraction
        print(f"One-way turnover: {turnover:.1%}")


def print_watchlist(f, held, n=5):                # next-best candidates that are not held
    cand = f[f["eligible"] & ~f.index.isin(held)].head(n)  # best eligible stocks we don't own
    if len(cand):
        print(f"\nWatchlist (next {len(cand)} eligible, not held)")
        for t, r in cand.iterrows():
            print(f"  {t:<7} eligible rank {int(r['erank']):>3}  1m {r['r1']:+.1%}  3m {r['r3']:+.1%}  6m {r['r6']:+.1%}")


def health_check(f, prev):                        # mid-cycle risk check of current holdings
    print("\nHealth check of current holdings")
    if prev is None or prev.empty:                # nothing to check
        print("  no stock holdings in the last review")
        return
    flagged = 0                                   # count of problem positions
    for t in prev.index:                          # look at each holding
        if t not in f.index or not bool(f.at[t, "valid"]):  # data missing
            print(f"  {t:<7} no valid data")
            flagged += 1
            continue
        r = f.loc[t]                              # that holding's current numbers
        flags = []                                # reasons for concern
        if not r["above_ma"]:
            flags.append("below 200-day average")
        if r["r6"] <= 0:
            flags.append("6-month return negative")
        if r["r1"] < -0.15:
            flags.append(f"1-month return {r['r1']:.0%}")
        if flags:                                 # print only the troubled ones
            print(f"  {t:<7} {'; '.join(flags)}")
            flagged += 1
    if flagged == 0:
        print("  all holdings still pass their checks")
    else:
        print("  (rules are applied at the next review; this is an early warning only)")


# ----------------------------------------------------------------------------
# BACKTEST
# ----------------------------------------------------------------------------
def backtest(close, vol, bench, uni, cfg):        # replay the system every 10 trading days
    sectors = uni["sector"]                       # sector lookup
    close = close.sort_index().ffill(limit=3)     # fill tiny gaps once
    bench = bench.reindex(close.index).ffill()    # align benchmark to the same dates
    use_mpt = cfg.mpt and HAVE_SCIPY          # MPT variants need scipy
    if cfg.mpt and not HAVE_SCIPY:            # tell the user why MPT lines are missing
        warn("scipy not installed: skipping the Modern Portfolio Theory strategies (pip install scipy)")
    kinds = {"mpt_sharpe": "maxsharpe", "mpt_minvar": "minvar"} if use_mpt else {}  # strategy -> optimiser (none without scipy / --no-mpt)
    idxs = list(range(MIN_HISTORY, len(close) - STEP, STEP))  # review dates (as row positions)
    rows = []                                     # one result row per holding period
    prev_w = {k: pd.Series(dtype=float) for k in ["momentum", *kinds]}  # last period's weights per strategy
    for i in idxs:                                # one iteration per review
        j = i + STEP                              # row position where this holding period ends
        feats = compute_features(close.iloc[: i + 1], vol.iloc[: i + 1])  # features using ONLY past data
        f = score_universe(feats, cfg)            # score and rank
        w, risk_on = choose_portfolio(f, sectors, prev_w["momentum"].index, bench.iloc[: i + 1], cfg)  # equal-weight portfolio
        book = {"momentum": w}                    # the strategies' target weights for this period
        if use_mpt and len(w) >= 3:               # same stocks, re-weighted by Markowitz using ONLY past data
            hist = close.iloc[max(0, i - MPT_DAYS): i + 1]  # trailing year up to the review date
            for key, kind in kinds.items():
                book[key] = mpt_weights(hist, list(w.index), kind, cfg.mpt_max_weight, cfg.rf) * w.sum()  # keep same total exposure
        else:                                     # no scipy / too few names: copy the equal-weight book
            for key in kinds:
                book[key] = w
        period = close.iloc[j] / close.iloc[i] - 1  # every stock's return over the holding period
        cash_ret = (1 + cfg.cash_yield) ** (STEP / 252) - 1  # interest earned by uninvested cash
        row = {"date": close.index[j], "invested": w.sum(), "names": len(w), "risk_on": risk_on}  # shared facts
        for key, wt in book.items():              # evaluate each strategy
            pr = period.reindex(wt.index).fillna(0)  # its holdings' returns
            gross = (wt * pr).sum() + (1.0 - wt.sum()) * cash_ret  # portfolio return + interest on cash
            allw = pd.concat([prev_w[key], wt], axis=1).fillna(0)  # old vs new weights
            trade = (allw.iloc[:, 1] - allw.iloc[:, 0]).abs().sum()  # fraction of the portfolio traded
            row[key] = gross - trade * cfg.cost_bps / 1e4  # return after trading costs
            if key == "momentum":                 # turnover is reported for the main strategy
                row["turnover"] = trade / 2
            prev_w[key] = wt                      # carry weights into the next period
        row["universe"] = period.reindex(f.index[f["valid"]]).mean()  # equal-weight average of ALL scored stocks
        rows.append(row)
    res = pd.DataFrame(rows).set_index("date")    # tidy results table
    pos = [close.index.get_loc(d) for d in res.index]  # row positions of period ends
    res["SPY"] = bench.iloc[pos].values / bench.iloc[[p - STEP for p in pos]].values - 1  # benchmark return, same periods
    return res                                    # per-period results


def stats(r):                                     # summary statistics for a return series
    r = r.dropna()                                # ignore gaps
    ppy = 252 / STEP                              # periods per year (about 25)
    eq = (1 + r).cumprod()                        # growth of 1 unit
    years = len(r) / ppy                          # length of the test in years
    return {"CAGR": eq.iloc[-1] ** (1 / years) - 1,  # compound annual growth rate
            "Vol": r.std() * np.sqrt(ppy),        # annualised volatility
            "Sharpe": r.mean() / r.std() * np.sqrt(ppy) if r.std() else np.nan,  # return per unit of risk
            "MaxDD": (eq / eq.cummax() - 1).min(),  # worst peak-to-trough fall
            "Worst2wk": r.min()}                  # worst single two-week period


def print_backtest(res, cfg):                     # display backtest results
    print(f"\nBacktest {res.index[0].date()} -> {res.index[-1].date()} "
          f"(review every {STEP} trading days, {cfg.cost_bps:.0f} bps cost per trade)")
    rows = {}                                     # one row per strategy
    for c in LABELS:                              # momentum, MPT variants, universe, SPY
        if c not in res.columns:                  # skip strategies that were not run
            continue
        s = stats(res[c])
        rows[LABELS[c]] = {"CAGR": f"{s['CAGR']:.1%}", "Vol": f"{s['Vol']:.1%}", "Sharpe": f"{s['Sharpe']:.2f}",
                           "MaxDD": f"{s['MaxDD']:.1%}", "Worst2wk": f"{s['Worst2wk']:.1%}"}
    print(pd.DataFrame(rows).T.to_string())       # print the comparison table
    print(f"\nAvg holdings: {res['names'].mean():.0f}   Avg invested: {res['invested'].mean():.0%}   "
          f"Avg one-way turnover per review: {res['turnover'].mean():.1%}   "
          f"Reviews with SPY below 200dma: {(~res['risk_on']).mean():.0%}")
    print("Survivorship bias applies (current index members only). MPT weights are re-fitted on PRIOR data only.")


# ----------------------------------------------------------------------------
# COMMAND LINE
# ----------------------------------------------------------------------------
def parse_args():                                 # define and validate all options
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=25, help="number of portfolio slots")
    ap.add_argument("--weights", type=float, nargs=3, default=[0.2, 0.4, 0.4],
                    metavar=("W1M", "W3M", "W6M"), help="blend weights for 1m, 3m, 6m z-scores")
    ap.add_argument("--buffer", type=float, default=1.3,
                    help="keep holdings ranked within top*buffer (1.0 = no buffer)")
    ap.add_argument("--max-sector", type=float, default=0.30, help="max share of names per sector")
    ap.add_argument("--max-weight", type=float, default=0.08, help="max weight per stock")
    ap.add_argument("--weighting", choices=["equal", "invvol"], default="equal",
                    help="equal = 1/N per slot (default); invvol = calmer stocks get more")
    ap.add_argument("--mpt", action=argparse.BooleanOptionalAction, default=True,
                    help="include Modern Portfolio Theory comparisons (needs scipy)")
    ap.add_argument("--mpt-max-weight", type=float, default=0.10, help="max weight per stock in MPT portfolios")
    ap.add_argument("--rf", type=float, default=0.04, help="risk-free rate (annual) used for Sharpe in MPT")
    ap.add_argument("--report", action=argparse.BooleanOptionalAction, default=True,
                    help="write the interactive HTML dashboard")
    ap.add_argument("--open", action="store_true", help="open the dashboard in your browser when done")
    ap.add_argument("--risk-adjust", action=argparse.BooleanOptionalAction, default=True,
                    help="divide returns by volatility before scoring")
    ap.add_argument("--winsor", type=float, default=3.0, help="cap z-scores at +/- this value")
    ap.add_argument("--min-dollar-vol", type=float, default=2e7,
                    help="minimum median daily traded value in $")
    ap.add_argument("--market-filter", action=argparse.BooleanOptionalAction, default=True,
                    help="cut exposure when SPY is below its 200-day average")
    ap.add_argument("--bear-exposure", type=float, default=0.5,
                    help="stock exposure when market filter triggers (0 = all cash)")
    ap.add_argument("--review-days", type=int, default=14, help="days between reviews")
    ap.add_argument("--force", action="store_true", help="record an OFFICIAL review now even if the 14 days are not up")
    ap.add_argument("--health-check", action="store_true", help="check current holdings between reviews")
    ap.add_argument("--dry-run", action="store_true", help="do not write any files")
    ap.add_argument("--today", help="YYYY-MM-DD override for the review-due check")
    ap.add_argument("--universe-file", help="CSV with a 'ticker' column (optional security, sector)")
    ap.add_argument("--no-refresh-universe", action="store_true", help="use the cached constituent list")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--years", type=int, default=6, help="history for the backtest")
    ap.add_argument("--cost-bps", type=float, default=10, help="cost per unit traded, in basis points")
    ap.add_argument("--cash-yield", type=float, default=0.0, help="annual yield on cash (backtest)")
    cfg = ap.parse_args()                         # read what the user typed
    if cfg.top < 5:                               # too few names is not a diversified portfolio
        ap.error("--top must be at least 5")
    if min(cfg.weights) < 0 or sum(cfg.weights) <= 0:  # blend weights must make sense
        ap.error("--weights must be non-negative and not all zero")
    total = sum(cfg.weights)                      # normalise so they sum to 1
    cfg.w1, cfg.w3, cfg.w6 = (w / total for w in cfg.weights)  # unpack into w1, w3, w6
    cfg.max_weight = max(cfg.max_weight, 1.0 / cfg.top)  # cap must allow a fully invested portfolio
    return cfg                                    # validated settings


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():                                       # program entry point
    cfg = parse_args()                            # read settings
    today = datetime.strptime(cfg.today, "%Y-%m-%d") if cfg.today else datetime.now()  # "today" for the schedule

    official = True                               # True = this run is recorded in the history (see below)
    next_due = None                               # date the next official review falls due
    if not cfg.backtest:                          # live path: check the 14-day schedule (no network yet)
        due, last, next_due = review_status(today, cfg)  # is an official review due?
        official = due or cfg.force               # due, or forced: record it. Otherwise: interim view.
        if not official:                          # NOT an error: you still get the full output
            print(f"Last official review: {last.date()}   Next due: {next_due.date()}   (today {today.date()})")
            print("INTERIM VIEW: showing today's portfolio against your last official review. "
                  "Nothing will be saved to the history.")

    uni = get_universe(cfg)                       # load the stock universe
    print(f"Universe: {len(uni)} stocks")
    years = cfg.years if cfg.backtest else 2      # backtests need more history than a live review
    close, vol = download_prices(list(uni.index) + ["SPY"], years)  # prices+volume for stocks and SPY
    bench = close.pop("SPY") if "SPY" in close.columns else pd.Series(dtype=float)  # separate the benchmark
    if "SPY" in vol.columns:                      # drop SPY volume too
        vol = vol.drop(columns="SPY")

    if cfg.backtest:                              # historical simulation path
        res = backtest(close, vol, bench, uni, cfg)  # run it
        print_backtest(res, cfg)                  # show it
        if not cfg.dry_run:                       # save the period-by-period results
            save_csv(res, DATA_DIR / "backtest_periods.csv")
            if cfg.report:                        # interactive dashboard
                path = build_backtest_report(REPORT_DIR, res, cfg, LABELS, STEP)
                print(f"\nInteractive dashboard: {path}")
                if cfg.open:
                    webbrowser.open(Path(path).resolve().as_uri())
        return

    close, vol = clean_data(close, vol)           # data quality gate
    asof = close.index[-1].strftime("%Y-%m-%d")   # date of the latest prices
    print(f"Prices as of {asof}")
    f = score_universe(compute_features(close, vol), cfg)  # features -> score -> rank
    label, prev, cash_old = latest_snapshot(asof) # last review (for comparison)

    prev_hold = prev.index if prev is not None else []  # tickers currently held
    w, risk_on = choose_portfolio(f, uni["sector"], prev_hold, bench, cfg)  # build the new target
    cash_new = 1.0 - float(w.sum())               # whatever is not invested is cash

    regime = "above" if risk_on else "BELOW"      # wording for the report
    note = ""                                     # extra explanation
    if cfg.market_filter:
        note = f" -> exposure cut to {cfg.bear_exposure:.0%}" if not risk_on else " -> fully allowed"
    elif not risk_on:
        note = " (market filter is off)"
    print(f"Stocks scored: {int(f['valid'].sum())}   Eligible: {int(f['eligible'].sum())}")
    print(f"Market regime: SPY {regime} its 200-day average{note}")

    port = w.to_frame().join(f[["rank", "erank", "r1", "r3", "r6", "vol", "score"]]).join(uni[["security", "sector"]])  # holdings + data
    port.index.name = "ticker"                    # name the index
    port["status"] = np.where(port["erank"] > cfg.top, "extended", "core")  # core = inside the top N eligible
    port = port.sort_values("weight", ascending=False)  # biggest positions first
    changes = compare(prev, w, uni, f, cfg, cash_old if prev is not None else 0.0, cash_new)  # add/remove report

    print_portfolio(port, cfg)                    # show target portfolio
    print_sector_summary(port, cash_new)          # show sector split and cash
    print_changes(changes, label)                 # show what to trade
    print_watchlist(f, set(w.index))              # show next-best candidates
    if cfg.health_check or not official:          # interim views always include the early-warning check
        _, held, _ = latest_snapshot(asof, inclusive=True)  # holdings of the newest official review
        health_check(f, held)                     # flags holdings that have broken their rules

    if cfg.dry_run:                               # preview only
        print("\n[dry-run] nothing saved")
        return
    if official:                                  # OFFICIAL review: record it in the history
        for kind in ("portfolio", "ranking"):     # protect anything a same-day re-run would overwrite
            p = SNAP_DIR / f"{kind}_{asof}.csv"
            if p.exists():
                ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy(p, ARCHIVE_DIR / f"{kind}_{asof}_{datetime.now():%H%M%S}.csv")  # timestamped backup
        cash_row = pd.DataFrame({"weight": [cash_new], "security": ["Cash"], "sector": ["Cash"]},
                                index=pd.Index([CASH], name="ticker"))  # cash stored as a row
        out = pd.concat([port, cash_row])         # holdings plus cash
        out.index.name = "ticker"
        save_csv(out, SNAP_DIR / f"portfolio_{asof}.csv")  # save the target portfolio
        ranked = f.join(uni[["security", "sector"]])  # the full ranking with company names
        ranked.index.name = "ticker"              # label the first column so the CSV reads back cleanly
        save_csv(ranked, SNAP_DIR / f"ranking_{asof}.csv")  # save the full ranking
        save_csv(changes, CHANGES_DIR / f"changes_{asof}.csv")  # save the change report
        print(f"\nSaved official review for {asof} in {DATA_DIR}/  (next review due "
              f"{(today + timedelta(days=cfg.review_days)).date()})")
    else:                                         # interim: history untouched
        print(f"\nInterim view only - history not changed. Next official review: {next_due.date()} "
              f"(add --force to record a review now).")

    if cfg.report:                              # interactive dashboard for this review
        names = uni["security"].to_dict()         # ticker -> company name (for tooltips)
        mpt_data = analyze(close, list(w.index), bench, cfg.mpt_max_weight, cfg.rf, names) if (cfg.mpt and len(w) >= 3) else None  # MPT tab data
        if cfg.mpt and not HAVE_SCIPY:
            warn("scipy not installed: the dashboard's Modern Portfolio Theory tab is skipped (pip install scipy)")
        path = build_review_report(REPORT_DIR, asof, cfg, f, port, changes, uni, close, bench,
                                       cash_new, risk_on, label, SNAP_DIR, CHANGES_DIR, mpt_data,
                                       official, next_due)
        print(f"Interactive dashboard: {path}")
        if cfg.open:                              # pop it open in the default browser
            webbrowser.open(Path(path).resolve().as_uri())

# ============================================================================
# MODERN PORTFOLIO THEORY (MPT)
# Re-weights the SAME stocks by mean-variance optimisation, to compare with equal weight. Needs scipy.
# ============================================================================


"""
MPT notes: expected returns are shrunk halfway toward the group average, the covariance matrix is shrunk toward
its diagonal, and weights are capped per stock. Weights fitted on a window and judged on the SAME window always
look better than real life; only the backtest (re-fit on past data, judged going forward) is out-of-sample.
"""
try:                                              # scipy provides the constrained optimiser
    from scipy.optimize import minimize           # SLSQP handles equality + bound constraints
    HAVE_SCIPY = True                             # remember that it is available
except Exception:                                 # not installed
    HAVE_SCIPY = False                            # callers check this and skip MPT gracefully

TD = 252                                          # trading days per year (annualisation factor)
MPT_DAYS = 252                                    # estimation window: about one year of daily returns


# ----------------------------------------------------------------------------
# ESTIMATION
# ----------------------------------------------------------------------------
def estimate(rets, shrink_mu=0.5, shrink_cov=0.3):  # expected returns and covariance, both shrunk
    mu_hist = rets.mean().values * TD             # historical average return, annualised
    mu = (1 - shrink_mu) * mu_hist + shrink_mu * mu_hist.mean()  # pull each mean toward the group mean
    s = rets.cov().values * TD                    # sample covariance, annualised
    s = (1 - shrink_cov) * s + shrink_cov * np.diag(np.diag(s))  # shrink toward the diagonal (less noisy)
    return mu, s, mu_hist                         # shrunk mu, shrunk covariance, raw mu


def port_stats(w, mu, s, rf):                     # expected return, volatility and Sharpe of a weight vector
    ret = float(w @ mu)                           # weighted average expected return
    vol = float(np.sqrt(max(w @ s @ w, 1e-16)))   # portfolio volatility from the covariance matrix
    return {"ret": ret, "vol": vol, "sharpe": (ret - rf) / vol}  # Sharpe = excess return per unit of risk


# ----------------------------------------------------------------------------
# OPTIMISERS
# ----------------------------------------------------------------------------
def _cap(n, cap):                                 # make sure the per-stock cap can actually sum to 100%
    return max(float(cap), 1.0 / n + 1e-9)        # e.g. 25 stocks need a cap of at least 4%


def _solve(obj, n, cap, extra=()):                # shared SLSQP call: weights in [0, cap], summing to 1
    c = _cap(n, cap)                              # effective cap
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}] + list(extra)  # fully invested (+ any extras)
    res = minimize(obj, np.full(n, 1.0 / n), method="SLSQP", bounds=[(0.0, c)] * n,
                   constraints=cons, options={"maxiter": 300, "ftol": 1e-10})  # start from equal weights
    if not res.success or not np.isfinite(res.x).all():  # solver failed: fall back to equal weights
        return np.full(n, 1.0 / n), False
    w = np.clip(res.x, 0, c)                      # remove tiny numerical violations
    return w / w.sum(), True                      # renormalise and report success


def min_variance(s, cap):                         # least risky long-only mix
    return _solve(lambda w: float(w @ s @ w), len(s), cap)[0]


def max_sharpe(mu, s, cap, rf):                   # highest return per unit of risk
    def neg_sharpe(w):                            # SLSQP minimises, so negate the Sharpe ratio
        v = np.sqrt(max(w @ s @ w, 1e-16))        # portfolio volatility
        return -(float(w @ mu) - rf) / v
    return _solve(neg_sharpe, len(s), cap)[0]


def frontier(mu, s, cap, points=25):              # lowest-risk portfolio for each target return
    n = len(mu)
    c = _cap(n, cap)
    w_min = min_variance(s, cap)                  # left-most point of the frontier
    lo = float(w_min @ mu)                        # its expected return
    left = c                                      # greedy fill: put the cap into the best-return names first
    w_hi, room = np.zeros(n), 1.0
    for k in np.argsort(-mu):                     # highest expected return first
        take = min(c, room)
        w_hi[k], room = take, room - take
        if room <= 1e-12:
            break
    hi = float(w_hi @ mu)                         # highest return reachable under the caps
    out = []                                      # (vol, ret) pairs
    for target in np.linspace(lo, hi, points):    # sweep target returns
        eq = {"type": "eq", "fun": lambda w, t=target: float(w @ mu) - t}  # hit this return exactly
        w, ok = _solve(lambda w: float(w @ s @ w), n, cap, extra=[eq])  # minimise variance subject to it
        if ok:
            out.append((float(np.sqrt(w @ s @ w)), float(w @ mu)))
    return out


def random_cloud(mu, s, count=1000, seed=7):      # random long-only mixes, to show the feasible region
    rng = np.random.default_rng(seed)             # fixed seed: same picture every run
    w = rng.dirichlet(np.full(len(mu), 0.6), size=count)  # random weights that sum to 1
    ret = w @ mu                                  # each mix's expected return
    vol = np.sqrt(np.einsum("ij,jk,ik->i", w, s, w))  # and volatility
    return vol, ret


def mpt_weights(close_window, tickers, kind, cap, rf):  # used by the backtest: re-fit weights on past data
    tickers = [t for t in tickers if t in close_window.columns]  # names with price data
    if len(tickers) < 3 or not HAVE_SCIPY:        # too few names (or no scipy): plain equal weights
        return pd.Series(1.0 / max(len(tickers), 1), index=tickers)
    rets = close_window[tickers].pct_change().iloc[1:]  # daily returns over the estimation window
    rets = rets.loc[:, rets.isna().mean() < 0.2].fillna(0.0)  # drop badly incomplete columns, fill small gaps
    if rets.shape[1] < 3 or len(rets) < 60:       # not enough usable data
        return pd.Series(1.0 / len(tickers), index=tickers)
    mu, s, _ = estimate(rets)                     # shrunk estimates
    w = min_variance(s, cap) if kind == "minvar" else max_sharpe(mu, s, cap, rf)  # chosen optimiser
    out = pd.Series(w, index=rets.columns)        # label the weights
    return out.reindex(tickers).fillna(0.0)       # names we dropped get zero


# ----------------------------------------------------------------------------
# FULL ANALYSIS FOR THE REVIEW DASHBOARD
# ----------------------------------------------------------------------------
def _basket(rets, w):                             # daily return of a fixed-weight (daily rebalanced) basket
    return (rets * w).sum(axis=1)


def _perf_stats(r):                               # realised stats of a daily return series
    eq = (1 + r).cumprod()
    return {"ret": float(eq.iloc[-1] - 1), "vol": float(r.std() * np.sqrt(TD)),
            "sharpe": float(r.mean() / r.std() * np.sqrt(TD)) if r.std() else np.nan,
            "maxdd": float((eq / eq.cummax() - 1).min())}


def analyze(close, tickers, bench, cap, rf, names=None):  # everything the dashboard's MPT tab needs
    if not HAVE_SCIPY:                            # cannot optimise without scipy
        return None
    tickers = [t for t in tickers if t in close.columns]
    window = close[tickers].iloc[-(MPT_DAYS + 1):]  # one year plus one day (for returns)
    rets = window.pct_change().iloc[1:]           # daily returns
    rets = rets.loc[:, rets.isna().mean() < 0.1].fillna(0.0)  # keep well-populated names
    if rets.shape[1] < 3:                         # need at least three stocks
        return None
    names_used = list(rets.columns)               # tickers actually analysed
    n = len(names_used)
    mu, s, mu_hist = estimate(rets)               # shrunk estimates (+ raw means)
    w_eq = np.full(n, 1.0 / n)                    # equal weight (what the system holds)
    w_mv = min_variance(s, cap)                   # minimum variance
    w_ms = max_sharpe(mu, s, cap, rf)             # maximum Sharpe
    weights = {"Equal weight (system)": w_eq, "Min variance (MPT)": w_mv, "Max Sharpe (MPT)": w_ms}  # candidates

    points = []                                   # the labelled dots on the risk/return chart
    for label, w in weights.items():
        p = port_stats(w, mu, s, rf)
        points.append({"name": label, "vol": p["vol"], "ret": p["ret"], "sharpe": p["sharpe"]})
    if len(bench.dropna()) > MPT_DAYS:            # SPY at its raw trailing numbers
        br = bench.dropna().pct_change().iloc[-MPT_DAYS:]
        points.append({"name": "SPY (trailing, raw)", "vol": float(br.std() * np.sqrt(TD)),
                       "ret": float(br.mean() * TD), "sharpe": float((br.mean() * TD - rf) / (br.std() * np.sqrt(TD)))})

    cv, cr = random_cloud(mu, s)                  # feasible-region cloud
    fr = frontier(mu, s, cap)                     # efficient frontier
    stocks = [{"ticker": t, "vol": float(np.sqrt(s[i, i])), "ret": float(mu[i])} for i, t in enumerate(names_used)]

    dates = [d.strftime("%Y-%m-%d") for d in rets.index]  # x-axis for the performance lines
    perf = {"dates": dates, "series": []}         # cumulative growth of 100 for each weighting
    for label, w in weights.items():
        r = _basket(rets, w)
        perf["series"].append({"name": label, "y": ((1 + r).cumprod() * 100).round(3).tolist(),
                               "stats": _perf_stats(r)})
    if len(bench.dropna()) > MPT_DAYS:
        b = bench.pct_change().reindex(rets.index).fillna(0.0)
        perf["series"].append({"name": "SPY", "y": ((1 + b).cumprod() * 100).round(3).tolist(),
                               "stats": _perf_stats(b)})

    corr = rets.corr().round(2)                   # correlation matrix of holdings
    rows = [{"ticker": t, "eq": float(w_eq[i]), "minvar": float(w_mv[i]), "maxsharpe": float(w_ms[i]),
             "name": (names or {}).get(t, t)} for i, t in enumerate(names_used)]  # side-by-side weights
    return {"points": points, "frontier": fr, "cloud": {"vol": cv[::2].round(4).tolist(), "ret": cr[::2].round(4).tolist()},
            "stocks": stocks, "perf": perf, "weights": rows,
            "corr": {"labels": names_used, "values": corr.values.tolist()},
            "avg_corr": float((corr.values.sum() - n) / (n * (n - 1))), "window_days": len(rets), "rf": rf, "cap": float(_cap(n, cap))}

# ============================================================================
# INTERACTIVE DASHBOARDS (HTML)
# Writes one self-contained .html file per review/backtest, with its own chart engine (no internet needed).
# ============================================================================

# ----------------------------------------------------------------------------
# JSON-SAFE CONVERSION
# ----------------------------------------------------------------------------
def _clean(o):                                    # recursively convert numpy/pandas values to plain JSON types
    if isinstance(o, dict):                       # dictionaries: clean every value
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):              # lists: clean every item
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):                 # arrays become lists
        return _clean(o.tolist())
    if isinstance(o, pd.Series):                  # Series become lists
        return _clean(o.tolist())
    if isinstance(o, (pd.Timestamp,)):            # dates become YYYY-MM-DD text
        return o.strftime("%Y-%m-%d")
    if isinstance(o, (np.bool_,)):                # numpy booleans
        return bool(o)
    if isinstance(o, (np.integer,)):              # numpy integers
        return int(o)
    if isinstance(o, (np.floating, float)):       # floats: NaN/inf are not valid JSON, so use null
        v = float(o)
        return round(v, 6) if math.isfinite(v) else None
    return o                                      # strings, ints, bools, None pass through


def _write_page(path, title, data):               # insert the data into the HTML template and save it
    payload = json.dumps(_clean(data), allow_nan=False, separators=(",", ":"))  # compact JSON
    payload = payload.replace("</", "<\\/")       # stop "</script>" inside data from ending the script early
    page = TEMPLATE.replace("__TITLE__", title.replace("<", "&lt;")).replace("__DATA__", payload)  # fill placeholders
    Path(path).parent.mkdir(parents=True, exist_ok=True)  # make sure the folder exists
    Path(path).write_text(page, encoding="utf-8") # write the finished page
    return path


# ----------------------------------------------------------------------------
# DATA ASSEMBLY: REVIEW
# ----------------------------------------------------------------------------
def _load_history(snap_dir, changes_dir):         # one row per saved official review
    rows = []                                     # collected results
    for i, p in enumerate(sorted(Path(snap_dir).glob("portfolio_*.csv"))):  # oldest to newest
        d = p.stem.replace("portfolio_", "")      # date from the file name
        df = pd.read_csv(p, index_col="ticker")   # the saved portfolio
        stocks = df.drop(index="CASH", errors="ignore")  # stock rows only
        cash = float(df.loc["CASH", "weight"]) if "CASH" in df.index else max(0.0, 1 - stocks["weight"].sum())
        turn = None                               # turnover unknown for the first review
        cp = Path(changes_dir) / f"changes_{d}.csv"  # that review's change report
        if i > 0 and cp.exists():
            ch = pd.read_csv(cp, index_col="ticker")
            turn = float(ch["change"].abs().sum() / 2)  # one-way traded fraction
        rows.append({"date": d, "holdings": int((stocks["weight"] > 0).sum()), "cash": cash, "turnover": turn})
    return rows


def _basic_perf(close, port, bench):              # fallback performance lines when the MPT tab is unavailable
    held = [t for t in port.index if t in close.columns]
    if not held or len(bench.dropna()) < 253:
        return None
    rets = close[held].pct_change().iloc[-252:].fillna(0.0)  # last year of daily returns
    r = rets.mean(axis=1)                         # equal-weight basket (daily rebalanced)
    b = bench.pct_change().reindex(rets.index).fillna(0.0)  # SPY on the same dates

    def st(x):                                    # realised statistics
        eq = (1 + x).cumprod()
        return {"ret": eq.iloc[-1] - 1, "vol": x.std() * np.sqrt(252),
                "sharpe": x.mean() / x.std() * np.sqrt(252) if x.std() else None,
                "maxdd": (eq / eq.cummax() - 1).min()}
    return {"dates": [d.strftime("%Y-%m-%d") for d in rets.index],
            "series": [{"name": "Equal weight (system)", "y": ((1 + r).cumprod() * 100).round(3).tolist(), "stats": st(r)},
                       {"name": "SPY", "y": ((1 + b).cumprod() * 100).round(3).tolist(), "stats": st(b)}]}


def build_review_report(report_dir, asof, cfg, f, port, changes, uni, close, bench,
                        cash, risk_on, label, snap_dir, changes_dir, mpt_data, official, next_due):
    names = uni["security"].to_dict()             # ticker -> company name
    sectors = uni["sector"]                       # ticker -> sector
    valid = f[f["valid"]].copy()                  # stocks that were scored
    valid["name"] = valid.index.map(lambda t: names.get(t, t))
    valid["sector"] = valid.index.map(lambda t: sectors.get(t, "Unknown"))
    held = set(port.index)                         # tickers in the target portfolio

    # --- table rows ---------------------------------------------------------
    holdings = []                                 # the portfolio, one dict per stock
    for t, r in port.sort_values("score", ascending=False).iterrows():
        px, ma = f.at[t, "price"], f.at[t, "ma200"]
        holdings.append({"t": t, "name": r["security"], "sector": r["sector"], "w": r["weight"], "rank": r["rank"],
                         "erank": r["erank"], "px": px, "vsma": px / ma - 1, "r1": r["r1"], "r3": r["r3"],
                         "r6": r["r6"], "vol": r["vol"], "score": r["score"], "status": r["status"]})
    universe = []                                 # every scored stock (for the map and ranking table)
    for t, r in valid.iterrows():
        universe.append({"t": t, "name": r["name"], "sector": r["sector"], "rank": r["rank"],
                         "erank": r["erank"] if r["eligible"] else None, "r1": r["r1"], "r3": r["r3"], "r6": r["r6"],
                         "vol": r["vol"], "score": r["score"], "vsma": r["price"] / r["ma200"] - 1,
                         "eligible": bool(r["eligible"]), "held": t in held})
    ch = changes[changes["action"].isin(["ADD", "REMOVE", "HOLD"])]  # change report rows (no cash line)
    chrows = [{"t": t, "name": r["security"], "sector": r["sector"], "action": r["action"], "old": r["old_weight"],
               "new": r["new_weight"], "chg": r["change"], "reason": r["reason"]} for t, r in ch.iterrows()]
    watch = f[f["eligible"] & ~f.index.isin(held)].head(15)  # next-best candidates
    watchrows = [{"t": t, "name": names.get(t, t), "sector": sectors.get(t, "Unknown"), "erank": r["erank"],
                  "r1": r["r1"], "r3": r["r3"], "r6": r["r6"], "vol": r["vol"], "score": r["score"]}
                 for t, r in watch.iterrows()]

    # --- sector statistics --------------------------------------------------
    g = valid.groupby("sector")                   # group the scored stocks by sector
    sec = pd.DataFrame({"r1": g["r1"].mean(), "r3": g["r3"].mean(), "r6": g["r6"].mean(),
                        "above": g["above_ma"].mean(), "elig": g["eligible"].mean(), "n": g.size()})
    sec["share"] = sec["n"] / sec["n"].sum()      # share of the universe
    sec["pw"] = port.groupby("sector")["weight"].sum().reindex(sec.index).fillna(0.0) if len(port) else 0.0  # portfolio weight
    sec = sec.sort_values("pw", ascending=False)  # biggest holding first
    sectors_data = {"labels": sec.index.tolist(), "pw": sec["pw"], "share": sec["share"], "r1": sec["r1"],
                    "r3": sec["r3"], "r6": sec["r6"], "above": sec["above"], "elig": sec["elig"], "n": sec["n"]}

    # --- market and performance ---------------------------------------------
    b = bench.dropna()
    market = None
    if len(b) >= 210:                             # SPY with its 200-day average
        ma = b.rolling(200).mean()
        bb, mm = b.iloc[-400:], ma.iloc[-400:]
        market = {"x": [d.strftime("%Y-%m-%d") for d in bb.index], "spy": bb.round(2), "ma": mm.round(2)}
    perf = mpt_data["perf"] if mpt_data else _basic_perf(close, port, bench)  # last-year growth lines

    # --- headline cards -----------------------------------------------------
    turnover = None if label is None else float(changes["change"].abs().sum() / 2)
    st = (perf["series"][0]["stats"] if perf else {})  # equal-weight basket stats
    cards = [
        {"k": "Market regime", "v": "RISK-ON" if risk_on else "RISK-OFF", "s": "SPY vs 200-day average", "cls": "good" if risk_on else "bad"},
        {"k": "Holdings", "v": f"{len(port)} / {cfg.top}", "s": "equal weight per slot" if cfg.weighting == "equal" else "inverse-vol weights"},
        {"k": "Cash", "v": f"{cash * 100:.1f}%", "s": "unfilled slots + market filter"},
        {"k": "Eligible / scored", "v": f"{int(f['eligible'].sum())} / {int(f['valid'].sum())}", "s": "pass every filter"},
        {"k": "Turnover", "v": "initial build" if turnover is None else f"{turnover * 100:.1f}%", "s": "one-way vs last official review"},
        {"k": "Basket 1Y return", "v": "n/a" if not st else f"{st['ret'] * 100:.1f}%", "s": "hypothetical, equal weight"},
        {"k": "Basket volatility", "v": "n/a" if not st else f"{st['vol'] * 100:.1f}%", "s": "last year, annualised"},
    ]
    if mpt_data:
        cards.append({"k": "Avg correlation", "v": f"{mpt_data['avg_corr']:.2f}", "s": "between holdings, last year"})

    data = {"mode": "review", "asof": asof, "official": bool(official),
            "next_due": next_due.strftime("%Y-%m-%d") if next_due is not None else None,
            "label": label, "settings": f"blend 1m/3m/6m = {cfg.w1:.0%}/{cfg.w3:.0%}/{cfg.w6:.0%} | top {cfg.top} | "
                                        f"{'equal weight' if cfg.weighting == 'equal' else 'inverse-vol'} | "
                                        f"market filter {'on' if cfg.market_filter else 'off'}",
            "cards": cards, "holdings": holdings, "universe": universe, "changes": chrows, "watch": watchrows,
            "sectors": sectors_data, "market": market, "perf": perf, "risk_on": bool(risk_on), "cash": cash,
            "history": _load_history(snap_dir, changes_dir), "mpt": mpt_data, "top": cfg.top}
    fname = f"report_{asof}.html" if official else f"interim_{asof}.html"  # interim views never overwrite an official report
    path = Path(report_dir) / fname
    return _write_page(path, f"Momentum {'review' if official else 'interim view'} {asof}", data)


# ----------------------------------------------------------------------------
# DATA ASSEMBLY: BACKTEST
# ----------------------------------------------------------------------------
BT_COLORS = {"momentum": "#2F6FDE", "mpt_sharpe": "#8E5FD0", "mpt_minvar": "#17A2B8",
             "universe": "#F58518", "SPY": "#6B7785"}  # fixed colour per strategy


def build_backtest_report(report_dir, res, cfg, labels, step):
    keys = [k for k in labels if k in res.columns]  # strategies that were actually run
    data = {"mode": "backtest", "dates": [d.strftime("%Y-%m-%d") for d in res.index],
            "keys": keys, "labels": {k: labels[k] for k in keys}, "colors": {k: BT_COLORS[k] for k in keys},
            "ret": {k: res[k] for k in keys}, "invested": res["invested"], "names": res["names"],
            "turnover": res["turnover"], "risk_on": res["risk_on"].astype(bool), "ppy": 252.0 / step,
            "settings": f"review every {step} trading days | {cfg.cost_bps:.0f} bps cost per trade | top {cfg.top} | "
                        f"{'equal weight' if cfg.weighting == 'equal' else 'inverse-vol'} | "
                        f"MPT: cap {cfg.mpt_max_weight:.0%}, rf {cfg.rf:.0%}, re-fit each review on the prior year"}
    return _write_page(Path(report_dir) / "backtest_report.html", "Momentum backtest", data)


# ----------------------------------------------------------------------------
# HTML / CSS / JAVASCRIPT TEMPLATE (the whole interactive front end)
# ----------------------------------------------------------------------------
TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#F5F7FA;--card:#fff;--ink:#1F2933;--muted:#667585;--line:#E2E7ED;--chip:#EEF2F7;--blue:#2F6FDE;--green:#2E8B57;--red:#C0392B;--amber:#B7791F}
@media(prefers-color-scheme:dark){:root{--bg:#0F141A;--card:#171E26;--ink:#E6EBF0;--muted:#94A3B3;--line:#2A3541;--chip:#1F2832;--green:#46B07A;--red:#E0594A;--amber:#E0A53A}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1240px;margin:0 auto;padding:22px 16px 70px}
h1{margin:0;font-size:24px}
.sub{color:var(--muted);margin:4px 0 14px}
.badge{display:inline-block;font-size:12px;font-weight:700;letter-spacing:.04em;padding:3px 9px;border-radius:99px;vertical-align:middle;margin-left:10px}
.badge.official{background:rgba(46,139,87,.15);color:var(--green)}.badge.interim{background:rgba(183,121,31,.16);color:var(--amber)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:10px;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 13px}
.card .k{color:var(--muted);font-size:11.5px;text-transform:uppercase;letter-spacing:.05em}
.card .v{font-size:21px;font-weight:700;margin-top:1px}.card .s{color:var(--muted);font-size:12px}
.good{color:var(--green)}.bad{color:var(--red)}
.tabs{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 14px;position:sticky;top:0;background:var(--bg);padding:8px 0;z-index:5}
.tab{border:1px solid var(--line);background:var(--card);color:var(--ink);padding:7px 14px;border-radius:8px;cursor:pointer;font:inherit}
.tab.on{background:var(--blue);border-color:var(--blue);color:#fff}
.panel{display:none}.panel.on{display:block}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:14px;margin-bottom:14px;align-items:start}
.grid.one{grid-template-columns:1fr}
@media(max-width:600px){.grid{grid-template-columns:1fr}}
.chart,.box{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px;min-width:0}
.chart-head{display:flex;justify-content:space-between;align-items:center;gap:8px}
.chart-title{font-weight:700}
.chart-note{color:var(--muted);font-size:12px;margin:2px 0 4px}
.chart-tools button,.btn{border:1px solid var(--line);background:var(--chip);color:var(--ink);border-radius:6px;padding:3px 9px;font:inherit;font-size:12px;cursor:pointer}
.chart-tools button.on,.btn.on{background:var(--blue);color:#fff;border-color:var(--blue)}
.legend{display:flex;flex-wrap:wrap;gap:4px 12px;margin:4px 0 2px}
.legend span{cursor:pointer;font-size:12px;display:inline-flex;align-items:center;gap:5px;user-select:none}
.legend i{width:11px;height:11px;border-radius:3px;display:inline-block}
.legend span.off{opacity:.35;text-decoration:line-through}
svg text{fill:var(--muted);font-size:11px}
svg .gl{stroke:var(--line);stroke-width:1}
svg .ax{stroke:var(--muted);stroke-width:1;opacity:.6}
svg .val{fill:var(--ink)}
.tip{position:fixed;display:none;pointer-events:none;z-index:50;background:#111a24;color:#f2f5f8;border-radius:8px;padding:7px 10px;font-size:12px;line-height:1.45;box-shadow:0 6px 20px rgba(0,0,0,.25);max-width:320px}
.tip b{color:#fff}
.tblbox{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin-bottom:14px}
.tblbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;gap:10px;flex-wrap:wrap}
.tblbar input{border:1px solid var(--line);background:var(--chip);color:var(--ink);border-radius:6px;padding:5px 9px;font:inherit;min-width:200px}
.tblscroll{overflow:auto;max-height:560px}
table.t{border-collapse:collapse;width:100%;font-size:13px}
.t th{position:sticky;top:0;background:var(--chip);text-align:left;padding:6px 9px;white-space:nowrap;cursor:pointer;user-select:none;z-index:1}
.t th.r,.t td.r{text-align:right}
.t td{padding:5px 9px;border-top:1px solid var(--line);white-space:nowrap}
.t tr:hover td{background:var(--chip)}
.t tr.ADD td:first-child{box-shadow:inset 4px 0 var(--green)}.t tr.REMOVE td:first-child{box-shadow:inset 4px 0 var(--red)}
.t .pos{color:var(--green)}.t .neg{color:var(--red)}.t .dim{color:var(--muted)}
h2{font-size:16px;margin:18px 0 8px}
.note{color:var(--muted);font-size:12.5px}
.controls{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;margin-bottom:14px}
.controls label{font-size:12px;color:var(--muted)}
.controls input[type=date]{border:1px solid var(--line);background:var(--chip);color:var(--ink);border-radius:6px;padding:3px 6px;font:inherit}
.chip{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);background:var(--card);border-radius:99px;padding:3px 11px 3px 8px;cursor:pointer;font-size:12.5px;user-select:none}
.chip i{width:11px;height:11px;border-radius:3px;display:inline-block}.chip.off{opacity:.4}
</style></head>
<body><div class="wrap" id="app"></div>
<script>
(function(){
"use strict";
const DATA = __DATA__;                                   // everything the page shows, injected by Python
const NS = "http://www.w3.org/2000/svg";                 // SVG namespace
const BLUE="#2F6FDE", GREEN="#2E8B57", RED="#C0392B", ORANGE="#F58518", PURPLE="#8E5FD0", TEAL="#17A2B8", GREY="#9AA5B1", INK="#6B7785";
const SECTOR_PAL=["#4C78A8","#F58518","#54A24B","#E45756","#72B7B2","#B279A2","#FF9DA6","#9D755D","#BAB0AC","#EECA3B","#8CD17D"];

// ---------- tiny helpers -----------------------------------------------------
function h(tag, props, ...kids){                         // build an HTML element
  const e=document.createElement(tag);
  for(const k in (props||{})){
    const v=props[k];
    if(k==="class") e.className=v;
    else if(k==="html") e.innerHTML=v;
    else if(k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if(v!=null) e.setAttribute(k, v);
  }
  for(const c of kids.flat()){ if(c==null||c===false) continue; e.append(c.nodeType?c:document.createTextNode(c)); }
  return e;
}
function sv(tag, props, parent){                         // build an SVG element
  const e=document.createElementNS(NS, tag);
  for(const k in (props||{})) if(props[k]!=null) e.setAttribute(k, props[k]);
  if(parent) parent.appendChild(e);
  return e;
}
const nz = v => v!=null && isFinite(v);                  // "is a real number"
const F = {                                              // number formatters
  pct:(d=1)=>v=>nz(v)?(v*100).toFixed(d)+"%":"–",
  spct:(d=1)=>v=>nz(v)?((v>=0?"+":"")+(v*100).toFixed(d)+"%"):"–",
  num:(d=2)=>v=>nz(v)?v.toFixed(d):"–",
  int:v=>nz(v)?String(Math.round(v)):"–",
  txt:v=>v==null?"":String(v),
};
const MON=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const T = s => Date.parse(s);                            // "YYYY-MM-DD" -> milliseconds
const fdate = t => new Date(t).toISOString().slice(0,10);
function dlabel(t, span){                                // axis label for a date
  const d=new Date(t);
  return span>1.4*365*864e5 ? MON[d.getUTCMonth()]+" "+d.getUTCFullYear() : MON[d.getUTCMonth()]+" "+d.getUTCDate();
}
function niceTicks(lo, hi, n){                           // "round number" tick positions
  if(!(hi>lo)){ lo-=1; hi+=1; }
  const raw=(hi-lo)/Math.max(n,1), p=Math.pow(10,Math.floor(Math.log10(raw))), m=raw/p;
  const step=(m<1.5?1:m<3?2:m<7?5:10)*p, out=[];
  for(let v=Math.ceil(lo/step)*step; v<=hi+step*1e-9; v+=step) out.push(+v.toFixed(12));
  return out;
}
function logTicks(lo, hi){                               // 1-2-5 ticks for a log axis
  const out=[]; const a=Math.floor(Math.log10(lo)), b=Math.ceil(Math.log10(hi));
  for(let e=a;e<=b;e++) for(const m of [1,2,5]){ const v=m*Math.pow(10,e); if(v>=lo*0.999&&v<=hi*1.001) out.push(v); }
  return out;
}
function bisect(arr, x){                                 // index of the element nearest to x in a sorted array
  let lo=0, hi=arr.length-1;
  while(hi-lo>1){ const mid=(lo+hi)>>1; if(arr[mid]<x) lo=mid; else hi=mid; }
  return Math.abs(arr[lo]-x)<=Math.abs(arr[hi]-x)?lo:hi;
}
function mix(c1,c2,t){                                   // blend two [r,g,b] colours
  return "rgb("+c1.map((v,i)=>Math.round(v+(c2[i]-v)*t)).join(",")+")";
}

// ---------- tooltip ----------------------------------------------------------
const tip=h("div",{class:"tip"}); document.body.append(tip);
function showTip(ev, html){
  tip.innerHTML=html; tip.style.display="block";
  const r=tip.getBoundingClientRect(); let x=ev.clientX+14, y=ev.clientY+14;
  if(x+r.width>innerWidth-8) x=ev.clientX-r.width-14;
  if(y+r.height>innerHeight-8) y=ev.clientY-r.height-14;
  tip.style.left=Math.max(4,x)+"px"; tip.style.top=Math.max(4,y)+"px";
}
function hideTip(){ tip.style.display="none"; }

// ---------- chart frame (title, tools, legend, responsive redraw) -----------
function frame(root, o, draw){
  const tools=h("div",{class:"chart-tools"});
  const legend=h("div",{class:"legend"});
  const body=h("div",{class:"chart-body"});
  const box=h("div",{class:"chart"}, h("div",{class:"chart-head"}, h("div",{class:"chart-title"},o.title||""), tools),
              o.note?h("div",{class:"chart-note"},o.note):null, legend, body);
  root.append(box);
  let lastW=0;
  const redraw=(force)=>{ const w=Math.floor(body.clientWidth); if(w<40) return; if(!force && w===lastW) return; lastW=w; body.innerHTML=""; draw(body,w); };
  new ResizeObserver(()=>redraw(false)).observe(body);   // draws when first shown and whenever resized
  return {box,tools,legend,body,redraw:()=>redraw(true)};
}
function buildLegend(fr, items, hidden, onToggle){       // clickable legend: hide/show series
  fr.legend.innerHTML="";
  items.forEach(it=>{
    const s=h("span",{class:hidden.has(it.name)?"off":"", onclick:()=>{ hidden.has(it.name)?hidden.delete(it.name):hidden.add(it.name); buildLegend(fr,items,hidden,onToggle); onToggle(); }},
              h("i",{style:"background:"+it.color}), it.name);
    fr.legend.append(s);
  });
}

// ---------- LINE CHART -------------------------------------------------------
// o: {title,note,series:[{name,color,x:[ms],y:[],dash,width}],yFmt,height,logToggle,rebase,hline,onZoom,onReset,range:[x0,x1]}
function lineChart(root,o){
  const st={x0:null,x1:null,log:!!o.log,hidden:new Set()};
  const fr=frame(root,o,draw);
  function sync(){ buildLegend(fr,o.series.filter(s=>o.legend!==false),st.hidden,()=>fr.redraw()); }
  function tools(){
    fr.tools.innerHTML="";
    if(o.logToggle) fr.tools.append(h("button",{class:st.log?"on":"",onclick:()=>{st.log=!st.log;tools();fr.redraw();}},"Log scale"));
    if(st.x0!=null || o.range) fr.tools.append(h("button",{onclick:()=>{ st.x0=st.x1=null; if(o.onReset) o.onReset(); tools(); fr.redraw(); }},"Reset zoom"));
  }
  function draw(body,W){
    const H=o.height||320, m={l:60,r:16,t:10,b:28};
    const rng=o.range||[st.x0,st.x1];
    const vis=o.series.filter(s=>!st.hidden.has(s.name));
    const allX=o.series.flatMap(s=>[s.x[0],s.x[s.x.length-1]]);
    const X0=Math.min(...allX), X1=Math.max(...allX);
    const d0=rng[0]??X0, d1=rng[1]??X1, zoomed=rng[0]!=null;
    const prep=vis.map(s=>{                               // points inside the window (optionally rebased to 100)
      const idx=[]; for(let i=0;i<s.x.length;i++) if(s.x[i]>=d0&&s.x[i]<=d1&&nz(s.y[i])) idx.push(i);
      const base=(o.rebase&&zoomed&&idx.length)?100/s.y[idx[0]]:1;
      return {s, pts:idx.map(i=>({i,x:s.x[i],y:s.y[i]*base}))};
    });
    const ys=prep.flatMap(p=>p.pts.map(q=>q.y));
    if(!ys.length){ body.append(h("div",{class:"note"},"No data in this range.")); return; }
    const useLog=st.log&&ys.every(v=>v>0);
    const tf=useLog?Math.log10:(v=>v);
    let lo=Math.min(...ys), hi=Math.max(...ys);
    if(o.hline!=null){ lo=Math.min(lo,o.hline); hi=Math.max(hi,o.hline); }
    let a=tf(lo), b=tf(hi); const pad=(b-a||1)*0.06; a-=pad; b+=pad;
    const sx=t=>m.l+(t-d0)/((d1-d0)||1)*(W-m.l-m.r);
    const sy=v=>m.t+(1-(tf(v)-a)/((b-a)||1))*(H-m.t-m.b);
    const svg=sv("svg",{width:W,height:H,viewBox:`0 0 ${W} ${H}`},body);
    const yt=useLog?logTicks(Math.pow(10,a),Math.pow(10,b)):niceTicks(a,b,5);
    let lastLab=null;                                     // skip ticks whose label would repeat (e.g. 25, 25, 25)
    yt.forEach(v=>{ const y=sy(v), lab=(o.yFmt||F.num(0))(v); sv("line",{x1:m.l,x2:W-m.r,y1:y,y2:y,class:"gl"},svg);
      if(lab!==lastLab) sv("text",{x:m.l-6,y:y+4,"text-anchor":"end"},svg).textContent=lab;
      lastLab=lab; });
    const nx=Math.max(2,Math.floor((W-m.l-m.r)/90));
    for(let i=0;i<=nx;i++){ const t=d0+(d1-d0)*i/nx;
      sv("text",{x:sx(t),y:H-8,"text-anchor":i===0?"start":i===nx?"end":"middle"},svg).textContent=dlabel(t,d1-d0); }
    if(o.hline!=null) sv("line",{x1:m.l,x2:W-m.r,y1:sy(o.hline),y2:sy(o.hline),class:"ax","stroke-dasharray":"4 3"},svg);
    prep.forEach(p=>{                                     // one path per series (breaks at gaps)
      let d="", pen=false;
      p.pts.forEach(q=>{ d+=(pen?"L":"M")+sx(q.x).toFixed(1)+" "+sy(q.y).toFixed(1); pen=true; });
      sv("path",{d,fill:"none",stroke:p.s.color,"stroke-width":p.s.width||2,"stroke-dasharray":p.s.dash||null,"stroke-linejoin":"round"},svg);
    });
    const guide=sv("line",{y1:m.t,y2:H-m.b,stroke:INK,"stroke-width":1,"stroke-dasharray":"3 3",visibility:"hidden"},svg);
    const dots=prep.map(p=>sv("circle",{r:4,fill:p.s.color,stroke:"#fff","stroke-width":1.5,visibility:"hidden"},svg));
    const sel=sv("rect",{y:m.t,height:H-m.t-m.b,fill:BLUE,opacity:.15,visibility:"hidden"},svg);
    const ov=sv("rect",{x:m.l,y:m.t,width:W-m.l-m.r,height:H-m.t-m.b,fill:"transparent",style:"cursor:crosshair"},svg);
    let dragX=null;
    const px=e=>{ const r=svg.getBoundingClientRect(); return e.clientX-r.left; };
    ov.addEventListener("mousemove",e=>{
      const x=px(e); const t=d0+(x-m.l)/(W-m.l-m.r)*(d1-d0);
      if(dragX!=null){ sel.setAttribute("x",Math.min(dragX,x)); sel.setAttribute("width",Math.abs(x-dragX)); sel.setAttribute("visibility","visible"); }
      const rows=[]; let gx=null;
      prep.forEach((p,k)=>{ if(!p.pts.length) return; const q=p.pts[bisect(p.pts.map(z=>z.x),t)];
        dots[k].setAttribute("cx",sx(q.x)); dots[k].setAttribute("cy",sy(q.y)); dots[k].setAttribute("visibility","visible");
        gx=gx==null?q.x:gx; rows.push({n:p.s.name,c:p.s.color,v:q.y}); });
      if(gx==null) return;
      guide.setAttribute("x1",sx(gx)); guide.setAttribute("x2",sx(gx)); guide.setAttribute("visibility","visible");
      rows.sort((u,v)=>v.v-u.v);
      showTip(e,"<b>"+fdate(gx)+"</b><br>"+rows.map(r=>`<span style="color:${r.c}">●</span> ${r.n}: <b>${(o.yFmt||F.num(2))(r.v)}</b>`).join("<br>"));
    });
    ov.addEventListener("mouseleave",()=>{ hideTip(); guide.setAttribute("visibility","hidden"); dots.forEach(d=>d.setAttribute("visibility","hidden")); dragX=null; sel.setAttribute("visibility","hidden"); });
    ov.addEventListener("mousedown",e=>{ dragX=px(e); });
    ov.addEventListener("mouseup",e=>{
      const x=px(e); const was=dragX; dragX=null; sel.setAttribute("visibility","hidden");
      if(was!=null && Math.abs(x-was)>8){
        const t0=d0+(Math.min(x,was)-m.l)/(W-m.l-m.r)*(d1-d0), t1=d0+(Math.max(x,was)-m.l)/(W-m.l-m.r)*(d1-d0);
        if(o.onZoom) o.onZoom(t0,t1); else { st.x0=t0; st.x1=t1; tools(); fr.redraw(); }
      }
    });
    ov.addEventListener("dblclick",()=>{ st.x0=st.x1=null; if(o.onReset) o.onReset(); else { tools(); fr.redraw(); } });
  }
  sync(); tools();
  return fr;
}

// ---------- BAR CHART --------------------------------------------------------
// o: {title,cats:[],series:[{name,color,values,colors}],horizontal,yFmt,labelFmt,tipFn(i),height}
function barChart(root,o){
  const hidden=new Set();
  const fr=frame(root,o,draw);
  if(o.series.length>1) buildLegend(fr,o.series,hidden,()=>fr.redraw());
  function draw(body,W){
    const ser=o.series.filter(s=>!hidden.has(s.name)), cats=o.cats, hz=!!o.horizontal;
    const ml=hz?Math.min(150,Math.max(60,Math.max(...cats.map(c=>String(c).length))*6.3+10)):56;
    const m={l:ml,r:hz?54:12,t:8,b:hz?26:(cats.length>8?58:30)};
    const H=o.height||(hz?Math.max(220,cats.length*(ser.length*13+7)+m.t+m.b):300);
    const vals=ser.flatMap(s=>s.values).filter(nz);
    let lo=Math.min(0,...vals), hi=Math.max(0,...vals); if(hi===lo){hi=lo+1;}
    const pad=(hi-lo)*0.04; if(lo<0) lo-=pad; hi+=pad;
    const svg=sv("svg",{width:W,height:H,viewBox:`0 0 ${W} ${H}`},body);
    const fmt=o.yFmt||F.num(1);
    const sc=hz?(v=>m.l+(v-lo)/(hi-lo)*(W-m.l-m.r)):(v=>m.t+(1-(v-lo)/(hi-lo))*(H-m.t-m.b));
    niceTicks(lo,hi,5).forEach(v=>{ const p=sc(v);
      if(hz){ sv("line",{x1:p,x2:p,y1:m.t,y2:H-m.b,class:"gl"},svg); sv("text",{x:p,y:H-9,"text-anchor":"middle"},svg).textContent=fmt(v); }
      else { sv("line",{x1:m.l,x2:W-m.r,y1:p,y2:p,class:"gl"},svg); sv("text",{x:m.l-6,y:p+4,"text-anchor":"end"},svg).textContent=fmt(v); } });
    const band=(hz?(H-m.t-m.b):(W-m.l-m.r))/cats.length, inner=Math.min(band*0.82,ser.length*(hz?18:34)), bw=inner/ser.length;
    cats.forEach((c,i)=>{
      const c0=(hz?m.t:m.l)+band*i+(band-inner)/2;
      ser.forEach((s,k)=>{ const v=s.values[i]; if(!nz(v)) return;
        const col=(s.colors&&s.colors[i])||s.color, z=sc(0), p=sc(v);
        const r=hz?sv("rect",{x:Math.min(z,p),y:c0+k*bw,width:Math.max(1,Math.abs(p-z)),height:Math.max(1,bw-1.5),fill:col,rx:2},svg)
                  :sv("rect",{x:c0+k*bw,y:Math.min(z,p),width:Math.max(1,bw-2),height:Math.max(1,Math.abs(p-z)),fill:col,rx:2},svg);
        if(o.labelFmt&&hz){ sv("text",{x:(v>=0?p+4:p-4),y:c0+k*bw+bw/2+3.5,"text-anchor":v>=0?"start":"end",class:"val"},svg).textContent=o.labelFmt(v,i); }
      });
      const tx=hz?sv("text",{x:m.l-6,y:c0+inner/2+4,"text-anchor":"end"},svg):
        sv("text",{x:c0+inner/2,y:H-m.b+14,"text-anchor":cats.length>8?"end":"middle",transform:cats.length>8?`rotate(-40 ${c0+inner/2} ${H-m.b+14})`:null},svg);
      tx.textContent=(!hz&&cats.length>14&&i%2)?"":(String(c).length>22?String(c).slice(0,21)+"…":c);  // thin out crowded x labels
      const hit=hz?sv("rect",{x:0,y:m.t+band*i,width:W,height:band,fill:"transparent"},svg):sv("rect",{x:m.l+band*i,y:0,width:band,height:H,fill:"transparent"},svg);
      hit.addEventListener("mousemove",e=>showTip(e,o.tipFn?o.tipFn(i):"<b>"+c+"</b><br>"+ser.map(s=>`<span style="color:${s.color}">●</span> ${s.name}: <b>${fmt(s.values[i])}</b>`).join("<br>")));
      hit.addEventListener("mouseleave",hideTip);
    });
    sv("line",hz?{x1:sc(0),x2:sc(0),y1:m.t,y2:H-m.b,class:"ax"}:{x1:m.l,x2:W-m.r,y1:sc(0),y2:sc(0),class:"ax"},svg);
  }
  return fr;
}

// ---------- SCATTER CHART ----------------------------------------------------
// o: {title,note,xLabel,yLabel,xFmt,yFmt,groups:[{name,color,r,opacity,labels,points:[{x,y,label,tip}]}],lines:[{name,color,pts:[[x,y]],dash,width}],height,zeroLines}
function scatterChart(root,o){
  const st={hidden:new Set(),x0:null,x1:null,y0:null,y1:null};
  const fr=frame(root,o,draw);
  const items=[...o.groups.map(g=>({name:g.name,color:g.color})),...(o.lines||[]).map(l=>({name:l.name,color:l.color}))];
  buildLegend(fr,items,st.hidden,()=>fr.redraw());
  function tools(){ fr.tools.innerHTML=""; if(st.x0!=null) fr.tools.append(h("button",{onclick:()=>{st.x0=st.x1=st.y0=st.y1=null;tools();fr.redraw();}},"Reset zoom")); }
  function draw(body,W){
    const H=o.height||400, m={l:62,r:16,t:10,b:42};
    const gs=o.groups.filter(g=>!st.hidden.has(g.name)), ls=(o.lines||[]).filter(l=>!st.hidden.has(l.name));
    const xs=[...gs.flatMap(g=>g.points.map(p=>p.x)),...ls.flatMap(l=>l.pts.map(p=>p[0]))].filter(nz);
    const ys=[...gs.flatMap(g=>g.points.map(p=>p.y)),...ls.flatMap(l=>l.pts.map(p=>p[1]))].filter(nz);
    if(!xs.length){ body.append(h("div",{class:"note"},"Nothing to show.")); return; }
    const ext=(a)=>{ const lo=Math.min(...a), hi=Math.max(...a), p=(hi-lo||1)*0.06; return [lo-p,hi+p]; };
    let [x0,x1]=ext(xs), [y0,y1]=ext(ys);
    if(st.x0!=null){ x0=st.x0; x1=st.x1; y0=st.y0; y1=st.y1; }
    const sx=v=>m.l+(v-x0)/(x1-x0)*(W-m.l-m.r), sy=v=>m.t+(1-(v-y0)/(y1-y0))*(H-m.t-m.b);
    const svg=sv("svg",{width:W,height:H,viewBox:`0 0 ${W} ${H}`},body);
    niceTicks(y0,y1,6).forEach(v=>{ sv("line",{x1:m.l,x2:W-m.r,y1:sy(v),y2:sy(v),class:"gl"},svg); sv("text",{x:m.l-6,y:sy(v)+4,"text-anchor":"end"},svg).textContent=(o.yFmt||F.num(2))(v); });
    niceTicks(x0,x1,8).forEach(v=>{ sv("line",{x1:sx(v),x2:sx(v),y1:m.t,y2:H-m.b,class:"gl"},svg); sv("text",{x:sx(v),y:H-m.b+15,"text-anchor":"middle"},svg).textContent=(o.xFmt||F.num(2))(v); });
    if(o.zeroLines){ if(x0<0&&x1>0) sv("line",{x1:sx(0),x2:sx(0),y1:m.t,y2:H-m.b,class:"ax"},svg); if(y0<0&&y1>0) sv("line",{x1:m.l,x2:W-m.r,y1:sy(0),y2:sy(0),class:"ax"},svg); }
    sv("text",{x:(m.l+W-m.r)/2,y:H-4,"text-anchor":"middle"},svg).textContent=o.xLabel||"";
    sv("text",{x:12,y:(m.t+H-m.b)/2,"text-anchor":"middle",transform:`rotate(-90 12 ${(m.t+H-m.b)/2})`},svg).textContent=o.yLabel||"";
    const clip=sv("clipPath",{id:"c"+Math.random().toString(36).slice(2)},svg); sv("rect",{x:m.l,y:m.t,width:W-m.l-m.r,height:H-m.t-m.b},clip);
    const g=sv("g",{"clip-path":`url(#${clip.id})`},svg);
    ls.forEach(l=>{ const d=l.pts.map((p,i)=>(i?"L":"M")+sx(p[0]).toFixed(1)+" "+sy(p[1]).toFixed(1)).join("");
      sv("path",{d,fill:"none",stroke:l.color,"stroke-width":l.width||2.5,"stroke-dasharray":l.dash||null},g); });
    const hits=[];
    gs.forEach(gr=>gr.points.forEach(p=>{ if(!nz(p.x)||!nz(p.y)) return;
      const cx=sx(p.x), cy=sy(p.y);
      sv("circle",{cx,cy,r:p.r||gr.r||4,fill:gr.color,opacity:gr.opacity??0.85,stroke:gr.stroke||"none","stroke-width":gr.stroke?1.5:0},g);
      if(gr.labels||p.showLabel){                          // label beside the dot; flip to the left near the right edge
        const left=cx>W-m.r-150, t=sv("text",{x:cx+(left?-1:1)*((p.r||gr.r||4)+3),y:cy+4+(p.dy||0),"text-anchor":left?"end":"start",class:"val",style:"font-size:11px;font-weight:600"},g);
        t.textContent=p.label; }
      hits.push({cx,cy,p,gr}); }));
    const ring=sv("circle",{r:8,fill:"none",stroke:INK,"stroke-width":2,visibility:"hidden"},svg);
    const sel=sv("rect",{fill:BLUE,opacity:.15,visibility:"hidden"},svg);
    const ov=sv("rect",{x:m.l,y:m.t,width:W-m.l-m.r,height:H-m.t-m.b,fill:"transparent",style:"cursor:crosshair"},svg);
    let drag=null; const pos=e=>{ const r=svg.getBoundingClientRect(); return [e.clientX-r.left,e.clientY-r.top]; };
    ov.addEventListener("mousemove",e=>{
      const [x,y]=pos(e);
      if(drag){ sel.setAttribute("x",Math.min(x,drag[0])); sel.setAttribute("y",Math.min(y,drag[1])); sel.setAttribute("width",Math.abs(x-drag[0])); sel.setAttribute("height",Math.abs(y-drag[1])); sel.setAttribute("visibility","visible"); }
      let best=null, bd=15*15; hits.forEach(q=>{ const d=(q.cx-x)**2+(q.cy-y)**2; if(d<bd){bd=d;best=q;} });
      if(best){ ring.setAttribute("cx",best.cx); ring.setAttribute("cy",best.cy); ring.setAttribute("visibility","visible");
        showTip(e,best.p.tip||("<b>"+best.p.label+"</b><br>"+(o.xLabel||"x")+": "+(o.xFmt||F.num(2))(best.p.x)+"<br>"+(o.yLabel||"y")+": "+(o.yFmt||F.num(2))(best.p.y))); }
      else { ring.setAttribute("visibility","hidden"); hideTip(); }
    });
    ov.addEventListener("mouseleave",()=>{ hideTip(); ring.setAttribute("visibility","hidden"); drag=null; sel.setAttribute("visibility","hidden"); });
    ov.addEventListener("mousedown",e=>{ drag=pos(e); });
    ov.addEventListener("mouseup",e=>{
      const [x,y]=pos(e), d=drag; drag=null; sel.setAttribute("visibility","hidden");
      if(d && Math.abs(x-d[0])>8 && Math.abs(y-d[1])>8){
        const nx0=x0+(Math.min(x,d[0])-m.l)/(W-m.l-m.r)*(x1-x0), nx1=x0+(Math.max(x,d[0])-m.l)/(W-m.l-m.r)*(x1-x0);
        const ny1=y0+(1-(Math.min(y,d[1])-m.t)/(H-m.t-m.b))*(y1-y0), ny0=y0+(1-(Math.max(y,d[1])-m.t)/(H-m.t-m.b))*(y1-y0);
        st.x0=nx0; st.x1=nx1; st.y0=ny0; st.y1=ny1; tools(); fr.redraw();
      }
    });
    ov.addEventListener("dblclick",()=>{ st.x0=st.x1=st.y0=st.y1=null; tools(); fr.redraw(); });
  }
  return fr;
}

// ---------- HEATMAP ----------------------------------------------------------
// o: {title,rows:[],cols:[],values:[[]],fmt,scale:'div'|'seq',vmax,vmin,cellText}
function heatmap(root,o){
  const fr=frame(root,o,draw);
  function color(v){
    if(!nz(v)) return "transparent";
    if(o.scale==="seq"){ const t=Math.max(0,Math.min(1,(v-(o.vmin??0))/((o.vmax??1)-(o.vmin??0)))); return mix([238,243,251],[47,111,222],t); }
    const t=Math.max(-1,Math.min(1,v/(o.vmax||1)));
    return t>=0?mix([236,239,243],[46,139,87],t):mix([236,239,243],[192,57,43],-t);
  }
  function draw(body,W){
    const R=o.rows.length, C=o.cols.length, top=o.colRotate?64:26, left=Math.min(130,Math.max(46,Math.max(...o.rows.map(r=>String(r).length))*6.6+8));
    const cw=(W-left-8)/C, ch=Math.max(o.cellH||20,0), H=top+R*ch+6;
    const svg=sv("svg",{width:W,height:H,viewBox:`0 0 ${W} ${H}`},body);
    o.cols.forEach((c,j)=>{ const x=left+cw*j+cw/2;
      const t=sv("text",{x,y:top-6,"text-anchor":o.colRotate?"start":"middle",transform:o.colRotate?`rotate(-55 ${x} ${top-6})`:null},svg); t.textContent=c; });
    o.rows.forEach((r,i)=>{ sv("text",{x:left-6,y:top+ch*i+ch/2+4,"text-anchor":"end"},svg).textContent=r;
      o.cols.forEach((c,j)=>{ const v=o.values[i][j];
        const rc=sv("rect",{x:left+cw*j,y:top+ch*i,width:Math.max(1,cw-1),height:Math.max(1,ch-1),fill:color(v),rx:2},svg);
        if(o.cellText&&cw>=30&&ch>=15){ const tt=sv("text",{x:left+cw*j+cw/2,y:top+ch*i+ch/2+4,"text-anchor":"middle",style:"fill:#1F2933;font-size:10.5px"},svg); tt.textContent=(o.fmt||F.num(2))(v); }
        rc.addEventListener("mousemove",e=>showTip(e,"<b>"+r+(o.rowsAreAlso?" × "+c:" · "+c)+"</b><br>"+(o.fmt||F.num(2))(v)));
        rc.addEventListener("mouseleave",hideTip); });
    });
  }
  return fr;
}

// ---------- SORTABLE / FILTERABLE TABLE -------------------------------------
// o: {title,cols:[{k,label,fmt,align,signed,cls}],rows:[],sort:[key,dir],rowClass,filter,maxRows}
function dataTable(root,o){
  const st={k:o.sort?o.sort[0]:null, dir:o.sort?o.sort[1]:-1, q:""};
  const tbody=h("tbody"), thead=h("thead"), table=h("table",{class:"t"},thead,tbody);
  const input=h("input",{type:"search",placeholder:"Filter…",oninput:e=>{st.q=e.target.value.toLowerCase();render();}});
  const count=h("span",{class:"note"});
  const box=h("div",{class:"tblbox"}, h("div",{class:"tblbar"},h("b",{},o.title||""),h("div",{},count," ",o.filter===false?null:input)), h("div",{class:"tblscroll"},table));
  root.append(box);
  function head(){
    thead.innerHTML=""; const tr=h("tr");
    o.cols.forEach(c=>tr.append(h("th",{class:c.align==="l"?"":"r",onclick:()=>{ if(st.k===c.k) st.dir=-st.dir; else {st.k=c.k; st.dir=(c.align==="l"?1:-1);} head(); render(); }},
      c.label+(st.k===c.k?(st.dir>0?" ▲":" ▼"):""))));
    thead.append(tr);
  }
  function render(){
    let rows=o.rows.slice();
    if(st.q) rows=rows.filter(r=>o.cols.some(c=>String(c.fmt?c.fmt(r[c.k],r):r[c.k]).toLowerCase().includes(st.q)));
    if(st.k){ rows.sort((a,b)=>{ const x=a[st.k], y=b[st.k]; if(x==null&&y==null) return 0; if(x==null) return 1; if(y==null) return -1;
      return (typeof x==="string"?x.localeCompare(y):x-y)*st.dir; }); }
    count.textContent=rows.length+" rows";
    const lim=o.maxRows||1000; tbody.innerHTML="";
    rows.slice(0,lim).forEach(r=>{
      const tr=h("tr",{class:o.rowClass?o.rowClass(r):""});
      o.cols.forEach(c=>{ const v=r[c.k]; let cls=c.align==="l"?"":"r";
        if(c.signed&&nz(v)) cls+=v>0?" pos":v<0?" neg":"";
        if(c.cls){ const z=c.cls(v,r); if(z) cls+=" "+z; }
        tr.append(h("td",{class:cls.trim()}, c.fmt?c.fmt(v,r):(v==null?"":v))); });
      tbody.append(tr); });
  }
  head(); render(); return box;
}

// ---------- page furniture ---------------------------------------------------
function cardsRow(cards){ return h("div",{class:"cards"},cards.map(c=>h("div",{class:"card"},h("div",{class:"k"},c.k),h("div",{class:"v "+(c.cls||"")},c.v),h("div",{class:"s"},c.s||"")))); }
function makeTabs(app,defs){
  const nav=h("div",{class:"tabs"}), panels=[];
  defs.forEach((d,i)=>{ const p=h("div",{class:"panel"+(i===0?" on":"")}); panels.push(p);
    nav.append(h("button",{class:"tab"+(i===0?" on":""),onclick:()=>{ [...nav.children].forEach((b,j)=>{b.classList.toggle("on",j===i);panels[j].classList.toggle("on",j===i);}); hideTip(); }},d.label)); });
  app.append(nav,...panels); return panels;
}
const grid=(p,one)=>{ const g=h("div",{class:"grid"+(one?" one":"")}); p.append(g); return g; };
const sectorColor=(()=>{ const names=[...new Set(((DATA.universe||[]).map(u=>u.sector)).concat((DATA.sectors?DATA.sectors.labels:[])))].sort(); const m={}; names.forEach((n,i)=>m[n]=SECTOR_PAL[i%SECTOR_PAL.length]); return s=>m[s]||GREY; })();

// =============================== REVIEW PAGE ================================
function buildReview(D){
  const app=$("#app");
  app.append(h("h1",{},"Momentum "+(D.official?"review":"interim view")+" · "+D.asof,
    h("span",{class:"badge "+(D.official?"official":"interim")},D.official?"OFFICIAL REVIEW":"INTERIM VIEW")));
  app.append(h("div",{class:"sub"},"Compared with: "+(D.label||"first review")+" | "+D.settings+
    (D.official?"":" | not saved to history; next official review "+D.next_due)));
  app.append(cardsRow(D.cards));
  const defs=[{label:"Overview"},{label:"Holdings"},{label:"Momentum map"}];
  if(D.mpt) defs.push({label:"Modern Portfolio Theory"});
  defs.push({label:"Market & sectors"},{label:"Changes & history"},{label:"Full ranking"});
  const P=makeTabs(app,defs); let pi=0;

  // --- Overview ---
  { const p=P[pi++], g=grid(p);
    const hd=D.holdings;
    barChart(g,{title:"Holdings ranked by momentum score",note:"bar = composite score; colour = sector; label = portfolio weight",
      cats:hd.map(r=>r.t),series:[{name:"Score",color:BLUE,values:hd.map(r=>r.score),colors:hd.map(r=>sectorColor(r.sector))}],horizontal:true,
      yFmt:F.num(1),labelFmt:(v,i)=>F.pct(1)(hd[i].w),
      tipFn:i=>{const r=hd[i];return `<b>${r.t}</b> ${r.name}<br>${r.sector}<br>weight <b>${F.pct(2)(r.w)}</b> · rank ${r.rank}<br>1M ${F.spct()(r.r1)} · 3M ${F.spct()(r.r3)} · 6M ${F.spct()(r.r6)}<br>vol ${F.pct()(r.vol)}`;}})
      .legend.append(...[...new Set(hd.map(r=>r.sector))].map(s=>h("span",{style:"cursor:default"},h("i",{style:"background:"+sectorColor(s)}),s)));  // sector colour key
    const rightCol=h("div"); g.append(rightCol); rightCol.style.display="grid"; rightCol.style.gap="14px";
    if(D.perf) lineChart(rightCol,{title:"Last year: these holdings vs SPY (rebased to 100)",
      note:"hypothetical, fixed weights, in-sample. Drag to zoom (re-bases at the start of the window).",rebase:true,yFmt:F.num(0),height:300,hline:100,
      series:D.perf.series.map((s,i)=>({name:s.name,color:[BLUE,PURPLE,TEAL,INK][i%4],dash:s.name==="SPY"?"5 3":null,width:s.name==="SPY"?2:2.4,x:D.perf.dates.map(T),y:s.y}))});
    if(D.market) lineChart(rightCol,{title:"Market regime: SPY vs 200-day average",note:D.risk_on?"SPY is ABOVE its 200-day average":"SPY is BELOW its 200-day average",
      logToggle:true,yFmt:F.num(0),height:240,series:[{name:"SPY",color:"#1F2933",x:D.market.x.map(T),y:D.market.spy},{name:"200-day average",color:BLUE,dash:"5 3",x:D.market.x.map(T),y:D.market.ma}]});
    const S=D.sectors;
    barChart(g,{title:"Sector allocation: portfolio vs universe",note:"cash: "+F.pct(1)(D.cash),cats:S.labels,
      series:[{name:"Portfolio",color:BLUE,values:S.pw},{name:"Universe (share of stocks)",color:GREY,values:S.share}],yFmt:F.pct(0)});
  }
  // --- Holdings ---
  { const p=P[pi++], hd=D.holdings;
    dataTable(p,{title:"Target portfolio ("+hd.length+" stocks)",sort:["score",-1],rowClass:r=>r.status,
      cols:[{k:"t",label:"Ticker",align:"l"},{k:"name",label:"Company",align:"l"},{k:"sector",label:"Sector",align:"l"},{k:"w",label:"Weight",fmt:F.pct(2)},
        {k:"rank",label:"Rank",fmt:F.int},{k:"erank",label:"Elig. rank",fmt:F.int},{k:"px",label:"Price",fmt:F.num(2)},{k:"vsma",label:"vs 200dma",fmt:F.spct(),signed:true},
        {k:"r1",label:"1M",fmt:F.spct(),signed:true},{k:"r3",label:"3M",fmt:F.spct(),signed:true},{k:"r6",label:"6M",fmt:F.spct(),signed:true},
        {k:"vol",label:"Vol",fmt:F.pct()},{k:"score",label:"Score",fmt:F.num(2)},{k:"status",label:"Status",align:"l",cls:v=>v==="extended"?"dim":""}],rows:hd});
    p.append(h("div",{class:"note"},"'extended' = outside the top "+D.top+" eligible: kept via the turnover buffer, or added because a sector cap blocked a higher-ranked stock."));
    const g=grid(p);
    heatmap(g,{title:"Holdings: 1M / 3M / 6M returns",rows:hd.map(r=>r.t),cols:["1M","3M","6M"],values:hd.map(r=>[r.r1,r.r3,r.r6]),
      fmt:F.spct(0),vmax:Math.max(.1,...hd.flatMap(r=>[r.r1,r.r3,r.r6].map(Math.abs)).sort((a,b)=>a-b).slice(0,Math.ceil(hd.length*2.7)).slice(-1)),cellText:true,cellH:20});
    barChart(g,{title:"Distance above 200-day average",cats:hd.map(r=>r.t),series:[{name:"vs 200dma",color:GREEN,values:hd.map(r=>r.vsma)}],horizontal:true,yFmt:F.pct(0),labelFmt:F.pct(0),height:Math.max(260,hd.length*20+40)});
  }
  // --- Momentum map ---
  { const p=P[pi++], U=D.universe, g=grid(p);
    const mk=(r)=>({x:r.r3,y:r.r6,label:r.t,tip:`<b>${r.t}</b> ${r.name}<br>${r.sector}<br>1M ${F.spct()(r.r1)} · 3M ${F.spct()(r.r3)} · 6M ${F.spct()(r.r6)}<br>vol ${F.pct()(r.vol)} · score ${F.num(2)(r.score)}<br>${r.held?"<b>HELD</b>":r.eligible?"eligible, not held":"not eligible"}`});
    const sets=[["Not eligible",U.filter(r=>!r.eligible),GREY,.55,3.5],["Eligible, not held",U.filter(r=>r.eligible&&!r.held),"#8FB1E8",.85,4.5],["Held",U.filter(r=>r.held),BLUE,.95,6.5]];
    scatterChart(g,{title:"Momentum map: 3-month vs 6-month return",note:"hover a dot; drag to zoom; click legend to hide groups",xLabel:"3-month return",yLabel:"6-month return",xFmt:F.pct(0),yFmt:F.pct(0),zeroLines:true,height:430,
      groups:sets.map(s=>({name:s[0],color:s[2],opacity:s[3],r:s[4],stroke:s[0]==="Held"?"#fff":null,labels:s[0]==="Held"&&s[1].length<=35,points:s[1].map(mk)}))});
    const mk2=(r)=>({x:r.vol,y:r.r6,label:r.t,tip:mk(r).tip});
    scatterChart(g,{title:"Risk vs reward: volatility vs 6-month return",note:"top-left is best: high return, low volatility",xLabel:"volatility (annualised, 3 months)",yLabel:"6-month return",xFmt:F.pct(0),yFmt:F.pct(0),zeroLines:true,height:430,
      groups:sets.map(s=>({name:s[0],color:s[2],opacity:s[3],r:s[4],stroke:s[0]==="Held"?"#fff":null,points:s[1].map(mk2)}))});
    dataTable(p,{title:"Watchlist: next best eligible stocks (not held)",filter:false,sort:["erank",1],
      cols:[{k:"t",label:"Ticker",align:"l"},{k:"name",label:"Company",align:"l"},{k:"sector",label:"Sector",align:"l"},{k:"erank",label:"Elig. rank",fmt:F.int},
        {k:"r1",label:"1M",fmt:F.spct(),signed:true},{k:"r3",label:"3M",fmt:F.spct(),signed:true},{k:"r6",label:"6M",fmt:F.spct(),signed:true},{k:"vol",label:"Vol",fmt:F.pct()},{k:"score",label:"Score",fmt:F.num(2)}],rows:D.watch});
  }
  // --- MPT ---
  if(D.mpt){ const p=P[pi++], M=D.mpt;
    p.append(h("div",{class:"box",style:"margin-bottom:14px"},h("b",{},"What is this? "),
      "The system holds these stocks equal-weighted. Modern Portfolio Theory asks whether the SAME stocks could have been weighted better using their past returns, volatility and correlations. "
      +"The dots below compare equal weight, the minimum-variance mix and the maximum-Sharpe mix against SPY and the efficient frontier. "
      +"Caution: optimisers are sensitive to noisy estimates, so expected returns are shrunk toward the group average and each stock is capped at "+F.pct(0)(M.cap)+". "
      +"The last-year lines are IN-SAMPLE (weights fitted on that same year) and look better than real life; the backtest page is the honest out-of-sample test."));
    const g=grid(p);
    const pc=["#1F2933",PURPLE,TEAL,ORANGE];
    const dys=[18,-8,4,4];                                // stagger labels so coinciding points stay readable
    const pts=M.points.map((q,i)=>({x:q.vol,y:q.ret,label:q.name,r:8,showLabel:true,dy:dys[i%4],tip:`<b>${q.name}</b><br>return ${F.pct(1)(q.ret)}<br>volatility ${F.pct(1)(q.vol)}<br>Sharpe ${F.num(2)(q.sharpe)}`}));
    scatterChart(g,{title:"Efficient frontier: same stocks, different weights",note:"x = risk, y = expected return (shrunk estimates; SPY at raw trailing values). Hover or drag-zoom.",xLabel:"volatility (annualised)",yLabel:"expected return (annualised)",xFmt:F.pct(0),yFmt:F.pct(0),height:470,
      groups:[{name:"Random mixes of these stocks",color:"#C9D3DF",r:2.6,opacity:.7,points:M.cloud.vol.map((v,i)=>({x:v,y:M.cloud.ret[i],label:"random mix",tip:`random mix<br>return ${F.pct(1)(M.cloud.ret[i])}<br>vol ${F.pct(1)(v)}`}))},
              {name:"Individual stocks",color:"#8FB1E8",r:4,opacity:.9,points:M.stocks.map(s=>({x:s.vol,y:s.ret,label:s.ticker,tip:`<b>${s.ticker}</b><br>return ${F.pct(1)(s.ret)}<br>vol ${F.pct(1)(s.vol)}`}))},
              ...M.points.map((q,i)=>({name:q.name,color:pc[i%4],r:8,opacity:1,stroke:"#fff",labels:true,points:[pts[i]]}))],
      lines:[{name:"Efficient frontier",color:RED,width:2.5,pts:M.frontier}]});
    const rightCol=h("div"); g.append(rightCol); rightCol.style.display="grid"; rightCol.style.gap="14px";
    lineChart(rightCol,{title:"Last year: equal weight vs MPT vs SPY (rebased to 100)",note:"in-sample (flattering to MPT). Drag to zoom.",rebase:true,yFmt:F.num(0),height:300,hline:100,
      series:M.perf.series.map((s,i)=>({name:s.name,color:pc[i%4],dash:s.name==="SPY"?"5 3":null,width:s.name.indexOf("Equal")===0?3:2,x:M.perf.dates.map(T),y:s.y}))});
    dataTable(rightCol,{title:"Last-year statistics (in-sample)",filter:false,
      cols:[{k:"name",label:"Strategy",align:"l"},{k:"ret",label:"Return",fmt:F.spct(),signed:true},{k:"vol",label:"Vol",fmt:F.pct()},{k:"sharpe",label:"Sharpe",fmt:F.num(2)},{k:"maxdd",label:"Max DD",fmt:F.pct(),signed:true}],
      rows:M.perf.series.map(s=>({name:s.name,...s.stats}))});
    dataTable(p,{title:"Weights: equal vs MPT",sort:["maxsharpe",-1],
      cols:[{k:"ticker",label:"Ticker",align:"l"},{k:"name",label:"Company",align:"l"},{k:"eq",label:"Equal weight",fmt:F.pct(2)},{k:"minvar",label:"Min variance",fmt:F.pct(2)},{k:"maxsharpe",label:"Max Sharpe",fmt:F.pct(2)}],rows:M.weights});
    const lab=M.corr.labels;
    heatmap(p,{title:"Correlation between holdings (last year of daily returns)",note:"darker = move together more. Average "+F.num(2)(M.corr.values.length?M.avg_corr:0)+". Low-correlation names reduce risk the most.",
      rows:lab,cols:lab,values:M.corr.values,fmt:F.num(2),scale:"seq",vmin:0,vmax:1,colRotate:true,cellText:true,cellH:Math.max(16,Math.min(28,Math.floor(900/lab.length))),rowsAreAlso:true});
  }
  // --- Market & sectors ---
  { const p=P[pi++], S=D.sectors, g=grid(p);
    if(D.market) lineChart(g,{title:"SPY vs its 200-day average",logToggle:true,yFmt:F.num(0),height:300,series:[{name:"SPY",color:"#1F2933",x:D.market.x.map(T),y:D.market.spy},{name:"200-day average",color:BLUE,dash:"5 3",x:D.market.x.map(T),y:D.market.ma}]});
    barChart(g,{title:"Average stock return by sector",cats:S.labels,series:[{name:"1M",color:"#9BB8E8",values:S.r1},{name:"3M",color:BLUE,values:S.r3},{name:"6M",color:"#1B3F8B",values:S.r6}],horizontal:true,yFmt:F.pct(0)});
    barChart(g,{title:"Breadth: % of each sector's stocks",note:"above 200-day average, and passing every filter",cats:S.labels,series:[{name:"above 200dma",color:GREY,values:S.above},{name:"eligible",color:GREEN,values:S.elig}],horizontal:true,yFmt:F.pct(0)});
    barChart(g,{title:"Sector weights in portfolio",cats:S.labels,series:[{name:"Weight",color:BLUE,values:S.pw,colors:S.labels.map(sectorColor)}],horizontal:true,yFmt:F.pct(0),labelFmt:F.pct(1)});
  }
  // --- Changes & history ---
  { const p=P[pi++], C=D.changes.filter(r=>r.action!=="HOLD"), g=grid(p);
    p.prepend(h("div",{class:"note",style:"margin-bottom:8px"},D.official?"Changes versus the previous official review.":"Changes versus your last official review (interim: nothing has been saved)."));
    if(C.length){ const rows=C.slice().sort((a,b)=>(b.action==="ADD"?b.new:-b.old)-(a.action==="ADD"?a.new:-a.old));
      barChart(g,{title:"Trades (green = buy, red = sell)",cats:rows.map(r=>r.t),series:[{name:"Weight",color:GREEN,values:rows.map(r=>r.action==="ADD"?r.new:-r.old),colors:rows.map(r=>r.action==="ADD"?GREEN:RED)}],horizontal:true,yFmt:F.pct(1),
        tipFn:i=>`<b>${rows[i].t}</b> ${rows[i].name}<br>${rows[i].action}<br>${rows[i].reason}`}); }
    const Hh=D.history;
    if(Hh.length>=2){ const xs=Hh.map(r=>T(r.date));
      lineChart(g,{title:"Holdings count over reviews",height:220,yFmt:F.num(0),series:[{name:"Holdings",color:BLUE,x:xs,y:Hh.map(r=>r.holdings)}]});
      lineChart(g,{title:"Cash % over reviews",height:220,yFmt:F.pct(0),series:[{name:"Cash",color:GREY,x:xs,y:Hh.map(r=>r.cash)}]});
      lineChart(g,{title:"One-way turnover per review",height:220,yFmt:F.pct(0),series:[{name:"Turnover",color:ORANGE,x:xs,y:Hh.map(r=>r.turnover)}]}); }
    else g.append(h("div",{class:"box note"},"History charts appear once you have two or more official reviews saved."));
    dataTable(p,{title:"All positions vs previous review",sort:["action",1],rowClass:r=>r.action,
      cols:[{k:"action",label:"Action",align:"l"},{k:"t",label:"Ticker",align:"l"},{k:"name",label:"Company",align:"l"},{k:"sector",label:"Sector",align:"l"},{k:"old",label:"Old weight",fmt:F.pct(2)},{k:"new",label:"New weight",fmt:F.pct(2)},{k:"chg",label:"Change",fmt:F.spct(2),signed:true},{k:"reason",label:"Reason",align:"l"}],rows:D.changes});
  }
  // --- Full ranking ---
  { const p=P[pi++];
    dataTable(p,{title:"Every scored stock",sort:["rank",1],rowClass:r=>r.held?"ADD":"",
      cols:[{k:"rank",label:"Rank",fmt:F.int},{k:"t",label:"Ticker",align:"l"},{k:"name",label:"Company",align:"l"},{k:"sector",label:"Sector",align:"l"},
        {k:"held",label:"Held",fmt:v=>v?"yes":"",align:"l"},{k:"eligible",label:"Eligible",fmt:v=>v?"yes":"no",align:"l",cls:v=>v?"":"dim"},
        {k:"r1",label:"1M",fmt:F.spct(),signed:true},{k:"r3",label:"3M",fmt:F.spct(),signed:true},{k:"r6",label:"6M",fmt:F.spct(),signed:true},
        {k:"vsma",label:"vs 200dma",fmt:F.spct(),signed:true},{k:"vol",label:"Vol",fmt:F.pct()},{k:"score",label:"Score",fmt:F.num(2)}],rows:D.universe});
  }
  app.append(h("p",{class:"note"},"Educational tool, not investment advice. Hypothetical results; momentum strategies can lose money."));
}

// ============================== BACKTEST PAGE ===============================
function buildBacktest(D){
  const app=$("#app");
  app.append(h("h1",{},"Momentum backtest"), h("div",{class:"sub"},D.dates[0]+" → "+D.dates[D.dates.length-1]+" | "+D.settings));
  const dates=D.dates.map(T), n=dates.length;
  const BT={from:dates[0],to:dates[n-1],on:new Set(D.keys)};
  const cardBox=h("div"), controls=h("div",{class:"controls"}), charts=h("div");
  app.append(cardBox,controls,charts);

  // ---- controls: date range + strategy chips ----
  const inFrom=h("input",{type:"date",value:fdate(BT.from),onchange:e=>{BT.from=Math.max(dates[0],T(e.target.value));render();}});
  const inTo=h("input",{type:"date",value:fdate(BT.to),onchange:e=>{BT.to=Math.min(dates[n-1],T(e.target.value));render();}});
  const yearsBtn=(label,yrs)=>h("button",{class:"btn",onclick:()=>{ BT.to=dates[n-1]; BT.from=yrs?Math.max(dates[0],BT.to-yrs*365.25*864e5):dates[0]; render(); }},label);
  controls.append(h("label",{},"From "),inFrom,h("label",{},"to "),inTo,yearsBtn("1Y",1),yearsBtn("2Y",2),yearsBtn("3Y",3),yearsBtn("All",0));
  D.keys.forEach(k=>controls.append(h("span",{class:"chip",onclick:e=>{ if(BT.on.has(k)&&BT.on.size>1) BT.on.delete(k); else BT.on.add(k); e.currentTarget.classList.toggle("off",!BT.on.has(k)); render(); }},
    h("i",{style:"background:"+D.colors[k]}),D.labels[k])));

  // ---- maths on the selected window ----
  function window(){ const ix=[]; for(let i=0;i<n;i++) if(dates[i]>=BT.from&&dates[i]<=BT.to) ix.push(i); return ix; }
  const mean=a=>a.reduce((s,v)=>s+v,0)/a.length;
  const sd=a=>{ const m=mean(a); return Math.sqrt(a.reduce((s,v)=>s+(v-m)*(v-m),0)/Math.max(1,a.length-1)); };
  function equity(r){ const e=[100]; r.forEach(v=>e.push(e[e.length-1]*(1+v))); return e; }
  function drawdown(e){ let pk=-Infinity; return e.map(v=>{ pk=Math.max(pk,v); return v/pk-1; }); }
  function stats(r,b){
    if(r.length<2) return {};
    const e=equity(r), dd=drawdown(e), yrs=r.length/D.ppy, s=sd(r);
    const out={cagr:Math.pow(e[e.length-1]/100,1/yrs)-1, vol:s*Math.sqrt(D.ppy), sharpe:s?mean(r)/s*Math.sqrt(D.ppy):null,
      maxdd:Math.min(...dd), best:Math.max(...r), worst:Math.min(...r), hit:r.filter(v=>v>0).length/r.length, growth:e[e.length-1]};
    if(b){ const mb=mean(b), vb=sd(b)**2; let cov=0; r.forEach((v,i)=>cov+=(v-mean(r))*(b[i]-mb)); cov/=Math.max(1,r.length-1);
      out.beta=vb?cov/vb:null; out.corr=(vb&&s)?cov/(Math.sqrt(vb)*s):null;
      const up=b.map((v,i)=>v>0?i:-1).filter(i=>i>=0), dn=b.map((v,i)=>v<0?i:-1).filter(i=>i>=0);
      out.upcap=up.length?mean(up.map(i=>r[i]))/mean(up.map(i=>b[i])):null; out.dncap=dn.length?mean(dn.map(i=>r[i]))/mean(dn.map(i=>b[i])):null; }
    return out;
  }
  const colorOf=k=>D.colors[k], keys=()=>D.keys.filter(k=>BT.on.has(k));

  function render(){
    inFrom.value=fdate(BT.from); inTo.value=fdate(BT.to);
    const ix=window(); charts.innerHTML=""; cardBox.innerHTML="";
    if(ix.length<4){ charts.append(h("div",{class:"box note"},"Pick a longer date range (at least 4 review periods).")); return; }
    const sub=a=>ix.map(i=>a[i]);
    const R={}; D.keys.forEach(k=>R[k]=sub(D.ret[k]));
    const spy=R["SPY"]; const S={}; D.keys.forEach(k=>S[k]=stats(R[k],k==="SPY"?null:spy));
    const x0=ix[0]>0?dates[ix[0]-1]:dates[0]-14*864e5, xs=[x0,...ix.map(i=>dates[i])];
    const ks=keys();

    // headline cards (first active strategy that is not SPY, else SPY)
    const main=ks.find(k=>k==="momentum")||ks[0], sm=S[main], sb=S["SPY"];
    cardBox.append(cardsRow([
      {k:"CAGR",v:F.pct(1)(sm.cagr),s:"SPY "+F.pct(1)(sb.cagr),cls:sm.cagr>=sb.cagr?"good":"bad"},
      {k:"Volatility",v:F.pct(1)(sm.vol),s:"SPY "+F.pct(1)(sb.vol)},
      {k:"Sharpe (rf=0)",v:F.num(2)(sm.sharpe),s:"SPY "+F.num(2)(sb.sharpe)},
      {k:"Max drawdown",v:F.pct(1)(sm.maxdd),s:"SPY "+F.pct(1)(sb.maxdd)},
      {k:"Growth of 100",v:F.num(0)(sm.growth),s:"SPY "+F.num(0)(sb.growth)},
      {k:"Avg turnover",v:F.pct(1)(mean(sub(D.turnover))),s:"one-way per review"},
      {k:"Avg invested",v:F.pct(0)(mean(sub(D.invested))),s:"holdings "+F.num(0)(mean(sub(D.names)))}]));
    cardBox.firstChild.insertAdjacentElement("beforebegin",h("div",{class:"note",style:"margin-bottom:6px"},"Cards show: "+D.labels[main]));

    const g1=grid(charts);
    const zoom=(t0,t1)=>{ BT.from=t0; BT.to=t1; render(); }, reset=()=>{ BT.from=dates[0]; BT.to=dates[n-1]; render(); };
    lineChart(g1,{title:"Growth of 100 (after trading costs)",note:"drag to zoom the whole page; double-click to reset",logToggle:true,yFmt:F.num(0),height:340,onZoom:zoom,onReset:reset,
      series:ks.map(k=>({name:D.labels[k],color:colorOf(k),dash:k==="SPY"?"5 3":null,width:k==="momentum"?3:2,x:xs,y:equity(R[k])}))});
    lineChart(g1,{title:"Drawdown from previous peak",height:340,yFmt:F.pct(0),onZoom:zoom,onReset:reset,
      series:ks.map(k=>({name:D.labels[k],color:colorOf(k),dash:k==="SPY"?"5 3":null,x:xs,y:drawdown(equity(R[k]))}))});
    const g2=grid(charts);
    const w=Math.max(2,Math.round(D.ppy));
    const roll=r=>{ const o=[]; for(let i=0;i<r.length;i++){ if(i+1<w){o.push(null);continue;} let p=1; for(let j=i+1-w;j<=i;j++) p*=1+r[j]; o.push(p-1); } return o; };
    if(ix.length>w) lineChart(g2,{title:"Rolling 1-year return",height:300,yFmt:F.pct(0),hline:0,onZoom:zoom,onReset:reset,
      series:ks.map(k=>({name:D.labels[k],color:colorOf(k),dash:k==="SPY"?"5 3":null,x:ix.map(i=>dates[i]),y:roll(R[k])}))});
    if(ks.includes("momentum")&&ks.length>1&&ks.some(k=>k!=="SPY"&&k!=="momentum")||ks.includes("SPY")&&ks.length>1) {
      const base=ks.includes("SPY")?"SPY":ks[0], em=equity(R[base]);
      lineChart(g2,{title:"Relative strength vs "+D.labels[base]+" (1.0 = same)",note:"rising = beating "+D.labels[base],height:300,yFmt:F.num(2),hline:1,onZoom:zoom,onReset:reset,
        series:ks.filter(k=>k!==base).map(k=>{ const e=equity(R[k]); return {name:D.labels[k],color:colorOf(k),x:xs,y:e.map((v,i)=>v/em[i])}; })}); }
    // calendar-year returns
    const byYear={}; ix.forEach((i,j)=>{ const y=new Date(dates[i]).getUTCFullYear(); (byYear[y]=byYear[y]||[]).push(j); });
    const years=Object.keys(byYear).sort();
    barChart(g2,{title:"Return by calendar year",note:"first and last years may be partial",cats:years,yFmt:F.pct(0),
      series:ks.map(k=>({name:D.labels[k],color:colorOf(k),values:years.map(y=>byYear[y].reduce((p,j)=>p*(1+R[k][j]),1)-1)}))});
    // risk / reward scatter
    scatterChart(g2,{title:"Risk vs return",note:"each dot = one strategy over the selected window",xLabel:"volatility (annualised)",yLabel:"CAGR",xFmt:F.pct(0),yFmt:F.pct(0),height:300,
      groups:ks.map(k=>({name:D.labels[k],color:colorOf(k),r:8,opacity:1,labels:false,points:[{x:S[k].vol,y:S[k].cagr,label:D.labels[k],tip:`<b>${D.labels[k]}</b><br>CAGR ${F.pct(1)(S[k].cagr)}<br>vol ${F.pct(1)(S[k].vol)}<br>Sharpe ${F.num(2)(S[k].sharpe)}`}]}))});
    // histogram of period returns
    const all=ks.flatMap(k=>R[k]), lo=Math.min(...all), hi=Math.max(...all), nb=18, bw=(hi-lo)/nb||0.01;
    const bins=[...Array(nb)].map((_,i)=>lo+bw*(i+.5));
    barChart(g2,{title:"Distribution of two-week returns",note:"how often each return size occurred",cats:bins.map(v=>(v*100).toFixed(1)+"%"),yFmt:F.num(0),
      series:ks.map(k=>({name:D.labels[k],color:colorOf(k),values:bins.map((_,i)=>R[k].filter(v=>Math.min(nb-1,Math.floor((v-lo)/bw))===i).length)}))});
    // exposure
    const g3=grid(charts);
    const xr=ix.map(i=>dates[i]);
    lineChart(g3,{title:"Invested % of portfolio",note:"drops when stocks fail the filters or SPY is below its 200-day average",height:260,yFmt:F.num(0),onZoom:zoom,onReset:reset,
      series:[{name:"Invested %",color:BLUE,x:xr,y:sub(D.invested).map(v=>v*100)}]});
    lineChart(g3,{title:"Number of holdings",height:260,yFmt:F.num(0),onZoom:zoom,onReset:reset,series:[{name:"Holdings",color:GREEN,x:xr,y:sub(D.names)}]});
    lineChart(g3,{title:"One-way turnover per review",height:260,yFmt:F.pct(0),onZoom:zoom,onReset:reset,series:[{name:"Turnover",color:ORANGE,x:xr,y:sub(D.turnover)}]});
    // statistics table
    const M=[["CAGR","cagr",F.pct(1),true],["Volatility","vol",F.pct(1)],["Sharpe (rf=0)","sharpe",F.num(2)],["Max drawdown","maxdd",F.pct(1),true],["Best 2 weeks","best",F.spct(1),true],["Worst 2 weeks","worst",F.spct(1),true],
             ["Positive periods","hit",F.pct(0)],["Growth of 100","growth",F.num(0)],["Beta vs SPY","beta",F.num(2)],["Correlation vs SPY","corr",F.num(2)],["Up-capture vs SPY","upcap",F.pct(0)],["Down-capture vs SPY","dncap",F.pct(0)]];
    const rows=M.map(m=>{ const r={metric:m[0]}; ks.forEach(k=>r[k]=S[k][m[1]]); r._f=m[2]; r._s=m[3]; return r; });
    const cols=[{k:"metric",label:"Metric",align:"l"},...ks.map(k=>({k,label:D.labels[k],fmt:(v,r)=>r._f(v),cls:(v,r)=>r._s&&nz(v)?(v>0?"pos":v<0?"neg":""):""}))];
    dataTable(charts,{title:"Statistics for the selected window",filter:false,cols,rows});
    charts.append(h("p",{class:"note"},"Strategies: 'Momentum (equal weight)' is the real system. The two MPT lines hold the SAME stocks but weights are re-fitted every review using only the prior year of data (min-variance or max-Sharpe, capped per stock). "
      +"'S&P 500 stocks, equal weight' averages every scored stock. SPY is the index fund. Costs are charged on every trade. Survivorship bias: only today's index members are used, so all results are optimistic. Educational tool, not investment advice."));
  }
  render();
}

function $(s){ return document.querySelector(s); }
if(DATA.mode==="review") buildReview(DATA); else buildBacktest(DATA);
})();
</script></body></html>
"""


if __name__ == "__main__":                        # run main() only when executed directly
    main()
