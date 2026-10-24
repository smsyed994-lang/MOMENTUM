# Momentum Dashboard: project guide

A small local web app that runs your momentum stock-picking scripts (NSE 500 and S&P 500) on demand
and shows the result in a browser, with notes, alerts, heatmaps and performance tracking.
Educational tool, not investment advice.

## 1. The big picture

```
  Yahoo Finance prices  ──►  momentum script (the "engine")  ──►  app.py (Flask server)  ──►  index.html (your browser)
  NSE / Wikipedia lists       scores + picks stocks               runs the engine, caches        tables, charts, buttons,
                                                                  prices, serves JSON            notes, settings
```

Three layers, each with one job:

| Layer | File(s) | Job |
|---|---|---|
| Engine | `NSE500_MOMENTUM_NOCAP.py`, `momentum_sectors.py` | The strategy itself: download data, score stocks, pick a portfolio, save reviews. Works on its own from the command line. **The app never edits these.** |
| Server | `app.py` | Imports an engine, runs it in the background, keeps prices cached, serves data and saves your settings and notes. |
| Page | `index.html` | The screen you look at. Draws everything from JSON that `app.py` sends. No other files needed. |

Plus `requirements.txt` (the Python packages to install).

## 2. File by file

### `NSE500_MOMENTUM_NOCAP.py`: the India engine
- Universe: the Nifty 500 list (tries niftyindices.com, then NSE archives, then your cached copy). Benchmark: Nifty 500 index.
- **Signal:** 1, 3 and 6-month returns, each turned into a z-score (how strong versus every other stock), capped at ±3, then blended 20% / 40% / 40%. Returns are divided by volatility first, so smooth climbers beat wild ones.
- **Filters:** liquid (traded value above Rs 5 crore), price above Rs 20, above its 200-day average, positive 6-month return.
- **Portfolio:** top N eligible stocks, equal weight, **no sector cap**. Existing holdings keep their place while they stay within a rank buffer, which cuts needless trading. Empty slots stay in cash. If the index is below its 200-day average, total exposure is cut (default to 50%).
- **Reviews:** every 14 days it saves holdings, the full ranking and an add/remove report with the reason for each change.
- Key functions: `get_universe`, `download_prices`, `clean_data`, `compute_features`, `score_universe`, `choose_portfolio`, `compare`, `latest_snapshot`, `read_snapshot`, `review_status`, `backtest`.
- Writes to the folder `momentum_data_nse500/`.

### `momentum_sectors.py`: the US engine
- Same idea for the S&P 500: SPY as the benchmark, a default **30% sector cap**, and a lower liquidity filter (about $20 million a day).
- Writes to the folder `momentum_data_biweekly/`, so the two markets never mix.
- Exposes the same function names as the India script, which is what lets one `app.py` drive both.

### `app.py`: the server (the glue)
What it does, in order:
1. **Loads engines.** Finds both scripts next to itself (ignoring upper/lower case in the name) and lets you switch between them. If one is missing it just offers the other.
2. **Loads prices with a cache** (`price_cache.pkl`). One full 2-year download about once a day, then only the last month is re-fetched. That's why later refreshes are fast.
3. **Runs the analysis** (`run_analysis`) in a background thread so the page never freezes. It calls the engine's own scoring and portfolio functions, then adds the app's extras:
   - rupee/dollar amounts and share counts from your capital
   - rank change versus the last review (Rank Δ)
   - sector concentration alerts
   - **performance since each review** (`compute_perf`), rebuilt from your saved review files and cached prices
4. **Saves official reviews** when you press the button, using the same file formats as the script.
5. **Schedules auto-refresh** (every 5 to 60 minutes, optionally only in market hours: IST for NSE, New York time for the US).
6. **Stores your notes, columns and settings** as JSON files, separately for each market.

Web endpoints (what the page talks to):

| Endpoint | Purpose |
|---|---|
| `GET /` | serves `index.html` |
| `GET /api/data` | the latest result |
| `GET /api/status` | running / progress / error / clock |
| `POST /api/refresh` | start a run (`{"record": true}` saves an official review) |
| `GET/POST /api/settings` | read or save settings |
| `GET/POST /api/notes`, `POST /api/columns` | notes and custom columns |
| `GET /api/markets`, `POST /api/market` | list markets, switch market |

### `index.html`: the page
One self-contained file: layout, styles and JavaScript together.
- **Header:** market dropdown, Refresh now, Save official review, Settings.
- **Banners and cards:** sector alert, rank-mover alert, market regime, holdings count, cash, turnover, review due.
- **Tabs:** Holdings, Changes, Watchlist, Sectors, Performance, Heatmap, Full ranking.
- **Every table:** click a header to sort, filter box (also searches your notes), a Notes column you type into, "+ Add column" for your own columns. Notes save when you click away.
- It polls `/api/status` every 3 seconds, and reloads the data when a run finishes.

### `requirements.txt`
`flask pandas numpy scipy yfinance requests lxml`. Install with `py -m pip install -r requirements.txt`.

## 3. Files the app creates (you don't write these)

```
momentum_data_nse500/            (and momentum_data_biweekly/ for the US)
  universe.csv                   cached stock list
  price_cache.pkl                cached prices (speeds up refreshes)
  webapp_settings.json           your settings for this market
  notes.json                     your notes and custom columns
  snapshots/portfolio_DATE.csv   each saved review's holdings + weights (+ cash row)
  snapshots/ranking_DATE.csv     each saved review's full ranking
  changes/changes_DATE.csv       add/remove report for that review
app_choice.json                  which market you used last (next to app.py)
```

**The `snapshots` folder is your track record.** Back it up. Performance tracking and Rank Δ are built from it. Deleting `price_cache.pkl` is harmless (it re-downloads).

## 4. Running it

```
cd "C:\Users\syed shuaib mohammed\Downloads\US STOCKS\momentum no sector"
py -m pip install -r requirements.txt     (first time only)
py app.py
```
Then open **http://127.0.0.1:5000** in the browser (never double-click `index.html`; it needs the server). Keep the command window open while you use it.

Typical routine: open the page → Refresh now → check alerts and Performance → on review day, press **Save official review** and place the trades from the Changes tab.

## 5. How the numbers in the page are built

- **Weight → rupees/dollars:** weight × your capital; shares = whole shares you can afford (US stocks need the FX rate in Settings).
- **Rank Δ:** rank at the last saved review minus today's rank. Positive means it moved up.
- **Sector alert:** fires when one sector's share of the portfolio reaches your threshold (default 40%).
- **Heatmap:** sector figures are the *median* stock in that sector; tiles show every stock coloured by the metric you pick.
- **Performance:** each review's target weights are assumed bought at that day's close and held to the next review (or today). Strategy return = sum of weight × stock return. Cash earns zero. No costs or taxes, so real results will be lower.

## 6. Things to know

- Yahoo data can be about 15 minutes late and occasionally wrong; the engine drops stocks with stale or short data and prints a warning.
- Backtests use today's index members, so they overstate results.
- Both scripts must stay in the same folder as `app.py`. If you rename one, update `ENGINE_MODULES` near the top of `app.py`.
- Switching markets is blocked while a refresh is running.
- The server only listens on your own computer (127.0.0.1). Don't expose it to the internet without adding a password.

## 7. Where to change things

| I want to... | Edit |
|---|---|
| Change the signal weights, filters, rank buffer | the engine script (`HORIZONS`, `score_universe`, `select_picks`) |
| Change the review interval | `--review-days` in the script (default 14) |
| Add a new setting to the page | `DEFAULT_SETTINGS` and `make_cfg` in `app.py`, plus the Settings box in `index.html` |
| Add a column or tab | the `COLS` / `TABS` lists in `index.html`, and the data in `run_analysis` in `app.py` |
| Add another market | add an entry to `ENGINE_MODULES` in `app.py` and a script with the same function names |
| Add email/Telegram alerts | after `run_analysis` finishes, in `_worker` in `app.py` |

## 8. Phone version (no PC needed)

The same page also runs as a standalone phone app. GitHub's free automation runs the engines on a schedule
(and when you tap Refresh), saves the results to the repo, and the phone page reads them.
See **PHONE_SETUP.md** for the one-time setup. Extra files used for this:

| File | Job |
|---|---|
| `export_results.py` | runs the analysis and writes `data/<market>.json` (what GitHub Actions executes) |
| `.github/workflows/refresh.yml` | the schedule and the "Refresh" trigger |
| `config.json` | engine settings used by scheduled runs (slots, sector cap...); the phone updates it |
| `manifest.webmanifest`, `sw.js`, `icon-*.png` | make the page installable on a home screen |
| `data/*.json` | latest results (created by the workflow) |

In this mode capital, notes and alert levels live only in the phone's browser, so they never reach GitHub.
