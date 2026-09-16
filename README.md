# Stock Pitch Screener

Finds long and short candidates among every US-listed company between $1B and
$20B market cap, using SEC filings (EDGAR) and Yahoo Finance prices. Built to
feed stock-pitch competitions: it looks for businesses whose economics are
turning (longs) and businesses that ran on something temporary and are priced
as if it were permanent (shorts).

**What it is good at:** surfacing names you would never have looked at, with a
tear sheet that tells you *why* it surfaced.

**What it is bad at:** ordering the top of the list. The #1 name is not better
than the #20 name. Treat every list as a pile of leads, not a ranking.

---

## 1. Install (once, ~10 minutes)

1. **Install Python** (3.11 or newer) from <https://www.python.org/downloads/>.
   On Windows, tick **"Add python.exe to PATH"** in the installer. On a Mac,
   the plain installer is fine.
2. **Download this repo**: green **Code** button on GitHub, **Download ZIP**,
   unzip it somewhere with **at least 6 GB of free disk space** (the SEC data
   is big). Or `git clone` it if you know git.
3. That's it. The launcher installs everything else on its first run.

## 2. Run it

**Windows:** double-click `run_screener.bat`.

**Mac / Linux:** open Terminal in the folder and run `bash run_screener.sh`.

The first run:

1. creates a private Python environment and installs the packages,
2. asks for your **name and email** (the SEC requires this on every bulk
   download; it goes to sec.gov only and is saved in `user_agent.txt`),
3. downloads ~3 GB of SEC data (10-40 minutes depending on your connection,
   resumable if it drops),
4. screens ~2,000 companies (60-90 minutes; it prints progress as it goes).

Leave the window open. Later runs skip steps 1-3 and take 30-60 minutes.

**Sanity check first (recommended):** run it with `--test` before committing
to the long run. It screens 12 companies in about 5 minutes and proves your
install and internet path work.

```
run_screener.bat --test          (Windows, from a terminal in the folder)
bash run_screener.sh --test      (Mac / Linux)
```

Test-mode output goes to `output/test_run/` and its scores are meaningless
(you cannot rank 12 companies against each other), so don't read them.

## 3. Read the results

Open **`output/SUMMARY.md`** first. It is a plain-text page (any editor, or
GitHub/VS Code render it nicely) with every list the screener produces, and
for each name: company, sector, market cap, score, **how many separate lists
it appears on**, the **mechanism tags** it fired, and a link to its tear sheet.

A sensible first hour:

1. Skim **Pitchable longs** and **Pitchable shorts**. Ignore the order.
2. Sort in your head by the **"also on" column**. A name that shows up on 3+
   lists has several independent reasons to exist. Start there.
3. Open the **tear sheet** (`output/tearsheets/TICKER_long.md` or
   `TICKER_short.md`). It shows every sub-score, the trailing 8 quarters of
   margins/revenue/cash flow, a survivability strip, and for shorts an
   **archetype verification checklist**: the specific things to confirm in the
   10-K/10-Q before you believe the tag.
4. Only then go read the filings. The screener is a reason to open a filing,
   never a substitute for it.

### The lists

Every list is a CSV in `output/` with all ~200 computed columns.
`universe_ranked.csv` has every scored company.

| Long side | what it is |
|---|---|
| `pitchable_longs.csv` | good business, temporary trough, simple story a judge can follow |
| `derated_compounder.csv` | proven moat, economics still intact, price or multiple has come down |
| `inflecting_thin_moat.csv` | margins / returns turning up, no moat required |
| `surviving_distressed_value.csv` | 35-80% drawdown, cheap, but green survivability and high F-score |
| `long_shortlist.csv`, `long_financials.csv` | highest overall long score; financials scored on their own metrics |

| Short side | what it is |
|---|---|
| `pitchable_shorts.csv` | overall short score with semis and biotech removed; `high_confidence` = 3+ mechanisms firing |
| `consumer_shorts.csv`, `industrials_shorts.csv` | sector cohorts ranked by **number of on-thesis mechanisms**, not score |
| `commodity_disconnect_shorts.csv` | commodity names the market is not pricing as cyclicals |
| `archetype_ran_on_temp_success.csv` | grew on something temporary (COVID demand, a price spike) and is priced as permanent |
| `ran_on_pricing_success.csv`, `pricing_masking_volume.csv` | growth came from price, volumes are falling |
| `priced_for_impossible_growth.csv`, `peer_multiple_disconnect.csv`, `losing_share_priced_rich.csv` | valuation vs. what the business can plausibly do |
| `capacity_decay.csv`, `ppi_windfall.csv` | asset turnover decaying; input-cost windfall not yet in the P&L |
| `short_shortlist.csv`, `short_financials.csv` | highest overall short score; financials on their own metrics |

### Things to know

- **Nothing here is a recommendation.** The tags are pattern matches on
  reported numbers. Roughly a third of them fall apart on reading the filing;
  that is the job.
- **Scores are percentiles within this run.** A score of 90 means "top 10% of
  the ~2,000 names on this measure today", not "90% likely to work".
- **Financial companies** (banks, insurers, REITs) use different metrics and
  live in their own files. Don't compare their scores to industrials.
- **Data freshness.** Filings appear a few days after they hit EDGAR, and only
  if they were in the SEC files when you downloaded them. Re-download every
  few weeks with `--refresh-data`.
- **Yahoo Finance hiccups.** A few tickers per run fail to download prices
  ("possibly delisted"). That is normal and they are simply skipped.

## 4. If something goes wrong

- **"Python was not found"**: reinstall Python with "Add to PATH" ticked, or
  on Mac install from python.org rather than relying on the system one.
- **The download stops**: just run it again. It resumes from where it stopped.
- **It ran out of memory / the laptop is struggling**: run with fewer workers,
  `run_screener.bat --workers 4`.
- **Anything else**: the full log of the last run is `output/run_<date>.log`.
  Send that to whoever gave you this.

You can also run the built-in checks to confirm the install is healthy
(no internet needed, ~1 minute):

```
.venv\Scripts\python -m pytest        (Windows)
.venv/bin/python -m pytest            (Mac / Linux)
```

## 5. What's in the folder

```
run_screener.bat / .sh   the launcher (creates .venv, installs, runs)
scripts/run_screener.py  what the launcher calls: checks, data, run, SUMMARY.md
scripts/download_data.py SEC bulk-file downloader (resumable)
scripts/clean.py         reclaims disk from old outputs / caches
screener/                the screener itself; every tunable is in config.py
tests/                   behavioral unit tests
docs/SCREENER_SPEC.md    the full methodology: every signal, weight, and why
data/                    SEC files + caches (downloaded, not in git)
output/                  results (generated, not in git)
```

If you want to change what it looks for, everything lives in
`screener/config.py` with the reasoning next to each number, and
`docs/SCREENER_SPEC.md` explains the design end to end.
