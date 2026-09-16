# Stock Pitch Competition Screener — Living Spec (v2)

> Status: **methodology locked pending first-run inspection.** v2 folds in all
> 38 verified findings from the 2026-07 audit (`METHODOLOGY_AUDIT.md` — full
> rationale and empirical verification live there; this doc records decisions),
> plus three user overrides logged 2026-07-07 (shortlist size, per-sector cap,
> commodity discount).
> Purpose: narrow a large universe down to a ranked list of long and short
> candidates that fit the *fingerprint* pitch judges reward. It is an idea
> generator for manual research, NOT a decision engine and NOT a backtester.

## Guiding philosophy
- Judges reward: **simple business model**, and for longs a **defensible moat**
  plus a **turnaround story**; for shorts a **false moat** and **temporarily
  inflated growth assumed to persist**.
- **Turnaround-first** on the long side (55/45). Distress is a *feature*,
  not a disqualifier — a financially distressed company with a curling-upward
  turnaround can be the most exciting long pitch.
- Feasibility (borrow availability, liquidity, public-vs-private) does **not**
  matter. No feasibility gates.
- **Permissive first**: no criterion disqualifies on its own; every criterion is
  a soft score contribution. Under-surfacing is the worse failure. Soft kernels,
  haircuts, and confidence multipliers are allowed; hard gates are not (the only
  hard filters are the universe definition and the shortlist hard cap).
- Process rule (audit 4.4): before coding, the metric→score kernel shape
  (direction, breakpoints, plateau/taper) must be documented for **every** row
  in both composites — "no hard gates" does not protect against a mis-shaped
  soft kernel.

## Data layer (chosen)
- **SEC EDGAR `companyfacts` XBRL API = workhorse.** 10+ yr standardized
  quarterly/annual line items, free. NOTE (corrected claim): standardization is
  at the *taxonomy* level, not concept-coverage level — every concept needs a
  tag-fallback ladder (see below). Prefer the **nightly bulk
  `companyfacts.zip` (~1.4 GB)** for the full-universe fetch; per-CIK API for
  incremental refresh.
- **SEC EDGAR `submissions` JSON = third source** (same host, same User-Agent
  rule): supplies `sic`/`sicDescription` (sector/industry grouping — NOT in
  companyfacts), form history (10-Q vs 20-F/40-F filer type, NT filings),
  8-K item codes (catalyst + forensic hooks), `formerNames` (de-SPAC
  detection), and accession numbers for filing links. Fetched and cached per
  CIK at roadmap step 3.
- **yfinance = market data only** (price history, market cap, shares, current
  multiples, 2–3 yr high, short interest, splits, FX pairs). Fundamentals there
  are shallow and not point-in-time — never used for financial-statement
  history. Prices fetched via chunked `yf.download` (chart endpoint), NOT
  per-ticker `.info` at universe scale (rate limits).
- Consequence: **current-snapshot idea generator, not a validated backtest**
  (survivorship bias + no clean point-in-time history). Accepted.
- EDGAR requires a **User-Agent header with contact info**: use
  your own name and email (see `user_agent.txt`). Respect the ~10 req/sec courtesy rate limit.

### Tag-fallback ladders (required; audit 1.2)
Every concept resolves through an ordered ladder with derive-from-components
rules. Core ladders (extend as coverage QA demands):
- Revenue: RevenueFromContractWithCustomerExcludingAssessedTax → Revenues →
  RevenueFromContractWithCustomerIncludingAssessedTax → SalesRevenueNet.
  Financials: RevenuesNetOfInterestExpense / InterestAndDividendIncomeOperating.
- Gross profit: GrossProfit, else revenue − (CostOfRevenue →
  CostOfGoodsAndServicesSold → CostOfGoodsSold + CostOfServices).
- SG&A: SellingGeneralAndAdministrativeExpense, else
  SellingAndMarketingExpense (or SellingExpense) + GeneralAndAdministrativeExpense.
- Interest expense: InterestExpense → InterestExpenseNonoperating →
  InterestAndDebtExpense (→ InterestIncomeExpenseNet with sign check).
- Total debt: LongTermDebtNoncurrent + LongTermDebtCurrent (else LongTermDebt)
  + ShortTermBorrowings/DebtCurrent + finance-lease liabilities (operating-lease
  tags excluded).
Ladder selection is **recency-aware and per-period** (pick the tag with
coverage inside the metric's actual window; stitch across the ~2018 ASC-606
revenue-tag switch). A metric is "missing" only after the full ladder fails.
**Required deliverable of roadmap step 4:** a data-coverage QA report — per
metric, % of universe with a usable value, split by sector tag.

## Quarterlyization module (roadmap step 3.5 — prerequisite; audit 1.1)
companyfacts does not serve discrete quarterly series for flow items. This
module constructs them:
1. Key facts by (taxonomy, tag, unit, start, end); dedupe keeping the latest
   `filed` (tie-break: accession number); restrict to 10-K/10-Q and /A forms.
2. **Never use `fy`/`fp` for period identification** (they describe the filing,
   not the fact — verified). Fiscal alignment from start/end dates + the
   company's fiscal year-end.
3. Classify durations: ~90d (discrete quarter), ~180d (H1 YTD), ~270d (9M YTD),
   ~360d (FY).
4. Derive Q4 income items = FY − (Q1+Q2+Q3). Derive quarterly cash-flow items
   by consecutive-YTD differencing (Q2 = H1−Q1, Q3 = 9M−H1, Q4 = FY−9M).
5. Validate derived quarters sum back to FY within tolerance; on failure
   **null the quarter, never interpolate**; enforce minimum point counts per
   window (nulls data points, not names).
6. Flag derived-Q4 observations; down-weight them in slope fits (or use
   Theil-Sen).
7. `restated` flag when the retained value differs >2–5% from the originally
   filed value (free forensic tell for the short side).
8. Staleness: flag names whose latest quarterly fact is >~200 days old.

### Derived series (audit 2.1)
Per metric, build from clean discrete quarters:
- **TTM** — rolling 4-quarter sums for flows; TTM-numerator/TTM-denominator for
  ratios. Used for **level** and **trough**.
- **YoY-delta** — Q_t − Q_{t−4}, fiscal-aligned. Used for **slope**
  (seasonality-free; slope-of-first-differences = the second derivative).
Raw-quarterly remains a tunable configuration; TTM/YoY is the default the
first shortlist is inspected on. FCF special case: never fit slopes to
discrete-quarter FCF — level/trough on TTM FCF, slope = YoY change in TTM FCF
margin.

## Universe
- US listings, market cap **$1B–$20B, core focus $1B–$10B**.
- **Financials & REITs included** with a sector tag; scored on the
  substitute-metric mapping (see Scoring foundations) and ranked in their own
  CSV section.
- Entity hygiene (audit 1.5): dedupe to one row per CIK (keep first-listed
  ticker — verified primary class lists first); use company_tickers_exchange.json
  and drop OTC rows; regex-drop derivative suffixes (-WT, -WS, -U, -UN, -RT,
  preferred patterns); detect non-operating entities **structurally** (no
  us-gaap facts in companyfacts, or recent forms N-CSR/N-PORT/N-2) — SIC is
  unreliable for funds. Tag-don't-drop BDCs/royalty trusts.
- Build (audit 1.4): shares from EDGAR
  (dei:EntityCommonStockSharesOutstanding → us-gaap:CommonStockSharesOutstanding
  /Issued → diluted WAS) × last close from chunked `yf.download`; widen the
  build-time band to ~$0.8B–$25B (dei shares can be a quarter stale), re-check
  caps with per-ticker `fast_info` only for the final ~150 names. Don't compute
  cap from dei shares for multi-class names — use primary ticker's yfinance
  marketCap.
- **Foreign private issuers** (audit 1.3): classify from submissions form
  history (20-F/40-F present, no 10-Q → `annual_only`; ifrs-full taxonomy →
  `ifrs`). Run the **annual-frequency inflection variant** (slope over 3–5 FYs,
  trough within 2 FYs) over an ifrs-full ladder. Check `unit` on every fact;
  FX-convert via yfinance pairs or skip the ratio — never mix currencies.
  Cap the weight of yfinance-only signals for annual_only names.

## Scoring foundations

### Normalization (audit 3.2)
All cross-sectional inputs are scored as **percentile ranks (0–100) vs the
current snapshot universe**; composites are weighted averages of ranks
("weighted average percentile vs today's universe" — snapshot-relative).
- Industry-structural level metrics (GM level/stability, margin-vs-industry)
  rank **within coarse sector buckets** (2–3 digit SIC / ownerOrg / yfinance
  sector; fallback to universe-wide when a bucket has <15 members).
- Documented exceptions: ROIC−WACC moat level blends an absolute component
  (is the spread positive?) with the sector-relative rank. EV/Sales richness on
  the short side is **universe-wide** (on-thesis: richly priced growth), logged
  as tunable. Self-normalized metrics stay as-is (inflection engine,
  vs-own-peak, drawdown, Piotroski/Beneish absolute scales).

### Missing-data policy (audit 3.1)
1. Derive before declaring missing (full tag ladder).
2. Missing/inapplicable = **excluded, weights renormalized** over computed
   metrics. Never zero-fill or neutral-fill. Multi-input composites rescale
   over applicable components.
3. Sector-driven applicability masks: GM/DIO/current-ratio components masked
   for banks/insurers; DIO only when inventory material (>~2% of revenue or
   assets); DSO only when receivables material and non-financial; DSO/DIO
   expansion measured YoY same-fiscal-quarter.
4. Thin-basis guards: per-name **coverage columns** (% of intended weight
   computed, per bucket, plus quarters-of-history); "low-confidence" tag below
   ~60% coverage; shrink bucket scores toward the universe median in proportion
   to missing weight.
5. Tear sheet lists missing/inapplicable metrics by name.

### Within-bucket weighting — theme level (audit 3.3)
Correlated metrics average (as rank percentiles) inside themes; themes carry
the weights. Starting points (user tunes):
- **Long turnaround bucket (55%):** profitability-inflection theme ~50%
  (six inflection metrics + op-margin-vs-peak), F-Score-rising ~15%, valuation
  reset (drawdown) ~20%, sentiment/squeeze (short interest × inflection) ~15%.
  (Insider buying removed from the universe-wide table — no data source; see
  Long composite.)
- **Long moat bucket (45%):** demonstrated-economics ROIC−WACC level,
  GM level+stability, earnings/FCF consistency — plus the new low-weight
  valuation row (~5–10% of the long composite, carved from moat).
- **Short:** false-moat theme, inflated-growth theme, single forensic theme
  {Beneish (intact), DSO/DIO expansion, Sloan accruals — computed once},
  confirmation (SI). Replace the "leverage/accruals" false-moat row with a real
  leverage-propping measure: ROE−ROA gap widening, or debt/EBITDA rising while
  ROIC falls.
After the first run: Spearman correlation matrix of all inputs; merge or
down-weight pairs >0.8.

### Financials/REITs substitute mapping (audit 3.4)
- **Banks:** skip WACC/ROIC entirely (Kd formula provably wrong — deposit
  interest in numerator, deposits outside debt tags); use **ROE − Ke** for both
  inflection and moat-level rows; efficiency ratio
  (NoninterestExpense / (InterestIncomeExpenseNet + NoninterestIncome)) replaces
  op-margin-vs-peak; F-Score drops current-ratio and gross-margin components,
  renormalized to 7.
- **Insurers:** ROE − Ke plus loss-ratio proxies where tagged
  (PolicyholderBenefitsAndClaimsIncurredNet / PremiumsEarnedNet), else reduced
  applicable set.
- **REITs:** FFO proxy = NI + D&A − gains (gain-tag fallback chain, else omit
  gains) for earnings inflection; replace the Altman-Z tag with interest
  coverage and debt/EBITDA.
- Beneish and Altman Z suppressed for financials/REITs (out-of-model).
- Financials/REITs ranked in their own section of the output.

### ROIC, WACC, and guards (audit 3.5, 3.6)
- **NOPAT** = TTM OperatingIncomeLoss × (1−t); t = TTM tax / TTM pretax,
  clamped [0%, 30%]; statutory ~21–25% fallback when pretax ≤ 0.
- **Invested capital** = total debt (excl. operating leases) + equity − cash;
  goodwill-inclusive for the moat test (goodwill-ex variant on tear sheet).
  Guard: if |IC| < 10% of assets, substitute EBIT/Assets and flag. Winsorize
  quarterly ROIC to ~[−50%, +50%]; winsorize the spread at 1st/99th percentiles.
- **Kd**: ladder above; computed only when debt > ~2% of assets (else D/V≈0);
  clamped [Rf+1%, Rf+8%]. Same t as NOPAT.
- **CAPM**: beta from 2-yr weekly returns vs S&P 500; Rf = current 10-yr; ERP
  ~4.5–5.5% single assumed figure; E = market cap, D = book debt.
- **Time dimension:** the turnaround row = **ROIC inflection** plus a snapshot
  "current ROIC−WACC < 0" boolean tag (under constant WACC the inflection is
  mathematically identical — spec'd honestly). The 5-yr moat spread varies at
  minimum the risk-free rate by year (^TNX history / FRED DGS10), beta and ERP
  constant; if skipped, the row is labeled "5-yr avg ROIC minus current WACC"
  and scored by percentile, not sign.
- Tear-sheet flag whenever any clamp/fallback fired.

## The inflection engine (core technical piece)
Turnaround signal = **second derivative**, not level. Per metric:
1. **Level** — percentile within the company's **full available TTM history**
   (~40+ quarters where available, capped ~40; verified free in the same JSON).
   Short histories: smooth shrinkage toward neutral 50 by n/(n+k), k≈8 — no
   hard minimum.
2. **Slope** — on the YoY-delta series, 3–6 quarter lookback (5–6q primary).
   Dollar metrics converted to margins first (NI/revenue, FCF/revenue, TTM
   revenue denominator floored at a small fraction of assets). Estimator:
   Theil-Sen blended with a **sign-consistency count** (fraction of positive
   YoY deltas). Unitless via division by the MAD of the company's own
   historical quarterly changes (floored/blended with cross-sectional median
   MAD), clipped ±3 — or the minimal-delta alternative: percentile of the
   current slope within the company's own history of rolling same-window slopes.
3. **Trough** — argmin of the TTM series over a trailing 8–12q reference
   window; contributes only if it occurred **≥2 quarters before the latest
   observation** AND cumulative recovery exceeds ~1× MAD of historical
   quarterly changes (a name still making lows scores 0 here, stays eligible
   via level/slope). Recency scored as a smooth ramp (max(0, 1 − q_since/8)),
   scaled by recovery magnitude capped at ~3 MADs. Slope anchored at the trough
   only when ≥3 post-trough points exist. The same construction, sign-inverted,
   is the short side's **peak-recency** primitive.
- **Combination within a metric: conjunctive** — components scaled to [0,1],
  geometric mean with per-component floors ~0.1–0.15 (soft, not a gate).
  Across metrics: additive.
- **Applied to six metrics:** ROIC(−WACC tag), gross margin, operating margin,
  net income (prefer operating income / continuing ops), FCF, **and revenue**.
- **Cost-cut haircut (audit 4.1):** a revenue-context classifier
  (growing / stabilizing / still-declining over 3–6q + longer window) applies a
  ~0.5–0.7× haircut to the margin/earnings/FCF inflection components ONLY when
  still-declining. "Cost-cut inflection" tear-sheet tag; quarterly revenue
  series always displayed.
- **One-off damping (audit 2.6):** breadth term — mean metric score ×
  sqrt(breadth/6); margin-led metric weights (GM 30 / OM 25 / ROIC 15 / NI 15 /
  FCF 15, revenue folded per tuning); robust outlier detection (k MADs from
  rolling median) — an outlier trough damps the trough component and prints an
  "outlier trough" flag; best-effort one-off-adjusted "core" series from a broad
  impairment/restructuring/settlement/valuation-allowance tag list (display +
  secondary score, never a filter). 5-yr peak constructs use median of top-3
  quarters, not max.
- **Structural breaks (audit 2.7):** per-signal (per-metric, per-company)
  minimum observation counts as soft confidence multipliers (level ≥12 obs,
  5-yr constructs ≥16; below, weight shrinks linearly). De-SPAC detection:
  submissions `formerNames` ~ /acquisition (corp|co)/i; >20× magnitude jumps
  near series start on NI and assets; own-history windows restart at the first
  post-break quarter. "Short history" tag when a metric's first datapoint is
  <3 years old.
- **As-of policy (audit 1.6):** windows are evaluated in **calendar quarters
  back from the run date**, not series positions; missing recent quarters count
  as elapsed time. Soft staleness decay once last-period-end is older than
  ~135–150 days (no compliant filer penalized). data-as-of, last-filed, and
  staleness-days are output columns; NT-filing tag surfaced.
- Windows (slope 3–6q, trough 4–6q on the reference defined above) remain
  **tunable**; re-tune the trough window on TTM data (TTM troughs lag ~1–2
  quarters).

## LONG — weighting 55 turnaround / 45 moat
| Theme | Metric | Signal wanted |
|---|---|---|
| Turnaround: profitability inflection (~50% of bucket) | Six inflection metrics (ROIC w/ spread<0 tag, GM, OM, NI, FCF, revenue) + op margin vs own 5-yr peak | depressed level + rising YoY slope + recovered trough; cost-cut haircut if revenue still declining |
| Turnaround: F-Score (~15%) | **TTM Piotroski**, quarterly-updated; "rising" = F(t)−F(t−4q), continuous credit | rising. Issuance signal: shares YoY growth >2% threshold (SBC passes, distressed raises fail). Financials: rescaled over available components |
| Turnaround: valuation reset (~20%) | Drawdown from **2–3 yr high** (not 52-wk) | monotone-then-plateau kernel: ~0 below 20%, full by ~35%, plateau through ~80%, gentle optional taper beyond. Small "price confirming" term (% above 2–3 yr low / positive 3-mo slope) |
| Turnaround: squeeze (~15%) | Short interest = sharesShort/sharesOutstanding, **multiplied by the inflection sub-score** (shortPercentOfFloat frequently None for small caps — display-only tear-sheet column, never scored or used as denominator; audit 4.7) | elevated SI only pays in proportion to turnaround evidence. dateShortInterest staleness stamp mandatory |
| Moat | ROIC − WACC level on the **best rolling 12-consecutive-quarter window in trailing ~7 yr ("demonstrated economics")**; trailing-5-yr version kept as a second column | high & positive; the demonstrated-vs-trailing spread = fallen-angel signature (sort key + tear sheet). False-moat cross-check guard (audit 4.2) — see guard row below |
| Moat | GM level + stability on the demonstrated window (or robust dispersion excl. trailing 6–8q) | high, stable; sector-relative rank |
| Moat | Earnings/FCF consistency | stable |
| Moat: valuation (~5–10% of composite) | Cheapness blend: EV/Sales percentile + **EV/normalized-EBIT percentile** (own-history MEDIAN op margin × current revenue — user 2026-07-08); P/TBV for financials | soft cheapness — breaks ties against fully re-rated recoveries. EV/Sales NOT EV/GP (GP depressed at the inflection); normalized-EBIT uses the MEDIAN margin so trough names read cheap on normal earnings, consistent with the same rationale |
| Context tags (no score weight) | Altman Z (or coverage/debt-EBITDA for REITs); **survivability traffic light** — cash runway (quarters, when FCF<0), near-term debt vs cash (LongTermDebtMaturities…NextTwelveMonths tag → DebtCurrent fallback → "n/a"), dilution trajectory (2-yr diluted-WAS growth, flag >15%) | distress + strong inflection + funded = narrative amplifier; distressed + <4q runway + debt>cash = "check financing first". Runway suppressed for financials |
| Guards on op-margin-vs-peak (audit 4.3) | decline-shape modifier (stumble vs erosion) + sector-relative peer check (≥8 peers else "no-peer-data") | sharp-drop-after-plateau credited; monotonic multi-year erosion ≈ 0; "secular/industry decline" tag when the sector fell in lockstep |
| Guard on demonstrated-economics moat (audit 4.2 cross-reference; no score weight) | Short-side **false-moat cross-check on the long tear sheet**: the name's false-moat theme score + its components, plus the demonstrated window's **start/end dates** (display contents canonical in Tear sheet Tier 1; components inherit the short side's applicability masks/coverage) | a demonstrated-peak window will crown COVID-spike/cyclical-top names as proven moats. Two catch mechanisms: (1) `false-moat-collision` tag = top-quartile **demonstrated-moat rank** AND top-quartile **false-moat theme score**, both vs the snapshot universe (financials within their own section), thresholds tunable; suppressed below the ~60% low-confidence coverage cutoff (reuses the missing-data policy) so thin-basis names can't fire it off one component — catches *still-elevated* collisions; (2) the **window dates** catch spikes too old for the peak-recency primitive (recency ramp reaches zero by ~8q; older peaks also fall outside the trailing 8–12q reference window). The cross-check and tag use the **raw pre-sector-haircut** false-moat theme score — the sector-wide-decline haircut applies only to the short composite, else cyclical tops (a named target) would suppress their own flag. Display + tag only, never a score change |

- Moat: keep all three core metrics; weakness in 1–2 does not disqualify.
- **Insider buying**: removed from the universe-wide table (companyfacts has no
  Form 4 data; yfinance insider tables don't scale). Shortlist-stage enrichment
  only: yfinance insider_purchases / netSharePurchaseActivity for the surfaced
  names (nullable). Roadmap upgrade: SEC quarterly Insider Transactions Data
  Sets or Form 4 XML parsing (code 'P' net of 'S').

## SHORT — false moat + inflated growth + forensics
| Theme | Metric | Signal wanted |
|---|---|---|
| False moat | ROIC/margins **peaked & rolling over** — peak-recency primitive (trough construction sign-inverted): high full-history percentile + negative YoY slope + recent recovered-from peak. **PLUS the at-peak SETUP variant (user 2026-07-08, VITL/CAKE archetype)**: strict product of (own-history level percentile ≥ ~80th) × (current 4q-change percentile within own history of 4q-changes ≥ ~60th — kills chronic margin expanders, whose current delta is typical for themselves) × (peak recency ≤ ~4q). Row = max(rollover, ~0.7 × at-peak). At-peak also opens the margin-vs-peer and run-up interaction gates | declining from a high, OR at an own-history extreme reached at an unusual rate (the setup, before the rollover) |
| False moat | Margin vs 3-digit-SIC peer median (fallback 2-digit, ≥8 members; SIC 6xxx excluded from margin peers) — **as an interaction term only**: contributes only alongside the name's own deteriorating trend or an active forensic flag | unsustainably high AND cracking. Unconditioned high margin contributes zero (else the row shorts every great business). "Structural reason?" = manual tear-sheet checklist line |
| False moat | Leverage propping: ROE−ROA gap widening, or debt/EBITDA rising while ROIC falls | yes |
| Inflated growth | Revenue deceleration: YoY growth per quarter over 8–12q; high base = peak YoY ≥ ~20–25% (or top quintile); decel = latest well below peak + negative 3–4q slope of the YoY series | yes. **M&A flag** (tag, not gate): Goodwill +>~20% YoY or acquisitions >~5% of TTM revenue — fake decel from lapping deals. Revenue-vs-GP-growth divergence = separate revenue-quality flag |
| Inflated growth | **Richness blend**: EV/Sales percentile (PEG dropped — yfinance field broken since ~6/2025, undefined for negative EPS, coverage-biased; EV = yfinance enterpriseValue → mktcap + EDGAR debt − cash; over EDGAR TTM revenue; universe-wide percentile) blended 50/50 with **EV/normalized-EBIT percentile** (user 2026-07-08): EV over (own full-history MEDIAN TTM op margin × current TTM revenue) — richness vs MEAN-REVERTED economics; peak-margin names look cheap on current earnings but rich here. NaN when median margin < ~0.5% or <12 margin obs | rich on sales AND/OR on normalized earnings. trailingPegRatio = sparse tear-sheet column only, never scored |
| Inflated growth: run-up (user 2026-07-08) | **The stock ran on temporary success**: run-up evidence = max(proximity to 2–3yr high: full within 10% of high, 0 beyond 35% below; 12-mo return: 0 at +30%, full at +80%) × fundamental-peak gate (max of rescaled rollover score and at-peak setup score) | at/near highs or big 12-mo run WHILE margins sit at an own-history extreme. A compounder at deserved highs scores ~0 (gate closed) |
| Inflated growth: pricing-masking-volume (user 2026-07-08 — the CAKE tell) | Conjunctive signature (geometric mean, no floors — all three legs required): GM expanding (+2pp/4q = full) × revenue YoY decaying (1.0 at ≤2%, 0 at ≥8%, hard 0 below −2% — outright decline is the deceleration row's territory) × asset turnover falling (−5%/4q = full) | price hikes propping revenue/margins over a shrinking volume base. `pricing_masking_volume` tear-sheet tag at ≥0.5 |
| Inflated growth | DSO / DIO expansion, YoY same-fiscal-quarter; DIO only where inventory material, DSO where receivables material & non-financial | rising |
| Inflated growth | Debt-funded buybacks: PaymentsForRepurchaseOfCommonStock > (OCF − capex) while total debt rises | yes — literally "growth assumed to persist" |
| Inflated growth | Dilution: YoY growth in split-adjusted diluted WAS | high (also optional long-side turnaround-quality tag) |
| Forensic (single theme; audit 3.3) | **Beneish M-Score**: continuous, percentile-ranked (no −2.22 binary), FY-over-FY; component fallbacks (DEPI: `Depreciation` else neutral; SGAI/GMI: ladders else neutral); indices winsorized [0.1, 10]; ≥6 of 8 computable else neutral; suppressed for financials/REITs; 8 components on tear sheet, SGI-dominance flagged | red flags |
| Forensic | **Sloan accruals = (TTM NI − TTM CFO) / avg assets** (cash-flow method — M&A-resistant; single definition shared with the false-moat line, computed once) | high |
| Forensic (event flags: tear-sheet red lines + small soft bump, not heavy weights) | `restated` flag; NT 10-K/10-Q in last ~2–3 yrs; 8-K item **4.01** (auditor change — review prompt, benign rotations exist) and **4.02** (non-reliance — high signal) | present |
| Confirmation (ADDITIVE BONUS only — user 2026-07-08) | Short interest — **hump-shaped** kernel (rising to moderate elevation, tapered when crowded), applied as a pure bonus of up to ~8 points ON TOP of the three-theme weighted composite. **Absence of SI is never a penalty**: no/missing SI = zero bonus, zero drag ("the market hasn't priced it in yet") | elevated-but-not-crowded SI supports/points to a short; its absence is not evidence against one. Framing: "consensus underappreciates how bad it is". The same SI feeds both composites — `on-both-lists` flag emitted at shortlist assembly (audit 4.7; see Shortlist section) |

- **Zero/low-revenue cohort (audit 5.5):** TTM revenue < $10M or revenue/mktcap
  < ~1% → tag; **null all sales-denominated short metrics** (EV/Sales, decel,
  margins, DSO/DIO, Beneish), renormalize, coverage % emitted; ranked in their
  own segmented section. Same nulling applies on the long side.

## Sector diversity & commodity treatment (user decisions 2026-07-07)
- **Soft per-sector cap** at shortlist assembly: with the ~100-name cap, start
  at ~12–15 names per sector group per side (tunable); overflow spills to
  next-ranked other-sector names; capped names remain visible in the full
  ranked CSV.
- **Commodity discount — RESCINDED (user 2026-07-08; superseded the
  2026-07-07 ×0.85 override):** under the ran-on-temporary-success system,
  commodity price-takers at cycle-top margins are prime shorts (HL/DK-type
  setups were being suppressed by the discount). No multiplier is applied on
  either composite. The `commodity` TAG remains on every row (SIC major
  groups: metal mining 10xx, coal 12xx, oil & gas extraction 13xx,
  nonmetallic minerals 14xx, paper/forest 24xx/26xx, commodity chemicals
  281x/286x/287x, petroleum refining 29xx, primary metals 33xx, water
  transport 44xx, agriculture 01xx–02xx) so the exposure stays visible and
  the discount is re-enableable per name in config.
- Tags: `sector-driven` (name's inflection matches its sector-group median move
  over the same window) and `cyclical` (high 10-yr margin volatility — reuses
  the moat stability calc; no bespoke episode counter).
- Short side: partial haircut to the false-moat contribution when the sector
  median margin is also declining (soft — preserves best-short-in-a-bad-sector).
  Haircut applies to the short-composite contribution only — the audit-4.2
  long-tear-sheet cross-check reads the raw pre-haircut theme score (see the
  LONG-table guard row).
- Sector grouping at 2-digit SIC / submissions `ownerOrg` / yfinance sector;
  fall back broader whenever a group has <~8 members.
- Sector-relative inflection blend (e.g. 60% own-history / 40% within-sector
  percentile) = v1.1 toggle, compared on/off before locking.

## Shortlist assembly & outputs (user override: surface MORE, not fewer)
- **Shortlist = all names clearing a tunable soft score threshold, hard-capped
  at ~100 per side.** Expectation: materially fewer than the cap qualify.
  (Supersedes the earlier ~20–40 target — surfacing more options is the goal;
  condensing happens manually.)
- Every output row carries BOTH composite scores + all theme/sub-scores +
  coverage columns + all tags — the long and short CSVs are not informationally
  separate.
- **`contested` flag** for names in the top decile of both composites —
  structurally likely (both screens fish the fallen/busted-growth pool) and
  pitch-relevant (turnaround-vs-value-trap is the debate judges care about).
  Distinct from the theme-level `false-moat-collision` tag (audit 4.2 guard
  row): that tag compares the demonstrated-moat theme against the false-moat
  theme at top-quartile thresholds and can fire on names nowhere near
  top-decile shorts overall — do not collapse the two.
- **`on-both-lists` flag (audit 4.7)** for names that clear the shortlist
  threshold on BOTH sides — structurally expected (the same SI feeds both
  composites, and both screens fish the same pool), and under the threshold +
  ~100-cap rule a name can make both shortlists without being top-decile on
  either. Threshold-based, **pre-cap**: a name spilled off one list by the
  hard or per-sector cap keeps the flag (the long/short tension is real
  regardless of cap mechanics). Independent of `contested` (neither implies
  the other: top-decile-both doesn't guarantee clearing the threshold, and
  vice versa); all three overlap flags (`on-both-lists`, `contested`,
  `false-moat-collision`) are separate columns.
- The full ranked universe is always also emitted, so widening past the
  threshold is trivial and boundary noise is visible.
- **`archetype_ran_on_temp_success` discovery cohort (user 2026-07-08):**
  a dedicated output section for names carrying the full CAKE/VITL
  signature REGARDLESS of composite rank — (pricing-masking-volume ≥ ~0.4
  OR at-peak setup ≥ ~0.5) AND run-up ≥ ~0.6 AND EV/normalized-EBIT
  richness ≥ ~70th pctile — sorted by signature strength (evidence ×
  normalized richness). Rationale: the composite stays honest when the
  accounting evidence is thin (CAKE's forensics are benign); the cohort is
  the archetype-first discovery list. Thresholds in config.ARCHETYPE_ROTS,
  tunable.

### Tear sheet (per surfaced name; audit 6.3)
- **Tier 1 (must-have):** composites + every theme/sub-score; per inflection
  metric the level/slope/trough triplet; the quarterly + TTM series per metric
  (pipeline retains per-metric intermediates); tags — sector, financials/REIT,
  commodity/cyclical/sector-driven, distress, survivability strip, cost-cut
  inflection, stumble-vs-erosion, outlier-trough, short-history, annual_only,
  contested, on-both-lists, restated, low-confidence, false-moat-collision;
  coverage % with
  missing metrics listed by name; guard/clamp flags (ROIC/WACC);
  **short-side false-moat cross-check (audit 4.2 guardrail; canonical display
  spec — the LONG-table guard row points here)** — on every long-side tear
  sheet, the false-moat theme score (raw, pre-sector-haircut; see guard row)
  and its three components (peaked-&-rolling-over, margin-vs-peer interaction,
  leverage propping) plus the demonstrated-economics window's start/end dates,
  rendered next to the moat rows. The data is already computed for every row
  (both composites are emitted universally); this is a **display requirement**
  so review catches COVID-spike/cyclical-top windows scored as proven moats.
  Components subject to the same applicability masks/coverage columns as on
  the short side.
- **Tier 2 (best-effort, nullable):** trailing ~8 quarters of 8-K item codes
  with legend (2.02 results, 5.02 exec change, 2.05 restructuring, 2.06
  impairment, 1.01 material agreement, 4.02 non-reliance, 3.01 delisting);
  next earnings date; insider transactions (yfinance, shortlist-stage);
  short interest + staleness stamp, plus shortPercentOfFloat (frequently None
  for small caps — display only, never scored; audit 4.7);
  NumberOfReportableSegments where tagged
  (display only); sector/SIC description, revenue scale, longBusinessSummary,
  direct EDGAR links to latest 10-K/10-Q — inputs for the **manual**
  simple-business judgment (segment count / geographic mix are NOT computable
  from companyfacts; judgment lives here, not in the score).

### First-run diagnostics (audit 6.4 — required before locking windows/weights)
1. Parameter sensitivity: slope {3,4,6}q × trough {4,6}q → top-50 Jaccard
   overlap AND full-universe Spearman correlation between combos.
2. Sub-score health: distribution, stdev, correlation with composite
   (detects dominating/dead sub-scores silently breaking the 55/45 split).
3. Trough-detector fire rate per window (>~half the universe firing = window
   can't discriminate).
4. Per-metric coverage counts, split by financials tag.
5. Top-decile sector mix vs universe mix (also calibrates the sector cap and
   commodity discount).
6. `false-moat-collision` fire rate in the long top decile / shortlist, plus
   the joint distribution of demonstrated-moat rank × false-moat theme rank
   (calibrates the audit-4.2 collision quartile thresholds before lock —
   near-zero firing = dead guardrail; >~half the shortlist firing = noise
   review will ignore).

### Exemplar-replay harness (validation; user 2026-07-08)
`scripts/replay.py <date> --tickers X,Y` reruns the whole screener AS OF a
past date: point-in-time facts via each observation's `filed` stamp (a
restatement filed later reverts to the originally-filed value), prices
truncated at the date, betas/drawdowns/run-ups on the truncated history,
market caps = as-of dei shares × as-of close, rf from the DGS10 history.
Purpose: calibrate against known good calls ("would the screen have caught
VITL in fall 2025?") — NOT a backtest. Documented limitations: today's
universe membership (survivorship-accepting; exemplars force-includable),
no historical short interest (SI rows absent uniformly, SI bonus 0 for
all), enterpriseValue via the mktcap+debt−cash fallback for all.

## Roadmap
1. Spec lock (this doc, v2). ← current
2. Universe builder — EDGAR tickers + entity hygiene + chunked prices →
   filtered, deduped, tagged list.
3. Data layer — bulk companyfacts + submissions JSON + yfinance, cached.
   3.5. **Quarterlyization module + derived TTM/YoY series.**
4. Metric library — tag ladders, ratios, composites, guards + the inflection
   engine. Deliverable includes the coverage QA report.
5. Two rankers — theme-weighted composites → ranked CSVs + tear sheets +
   first-run diagnostics.
6. Inspect diagnostics → tune windows/weights/threshold → lock v1 shortlists.
7. (Later) optional dashboard; Form 4 insider upgrade.

## Resolved decisions (log)
- Inflection: slope lookback 3–6q, trough window 4–6q, both tunable. (v1)
- WACC: full CAPM build. (v1) — v2 adds guards, constant-WACC treatment for
  inflection, historical Rf for the moat spread.
- Moat: keep all three metrics; weakness in 1–2 doesn't disqualify. (v1)
- Financials/REITs: include with sector tag. (v1) — v2 operationalizes with
  substitute metrics + segmented output.
- Permissive: prefer too many over too few. (v1, reaffirmed)
- **2026-07-07 (audit fold-in + user overrides):**
  - All 38 audit findings adopted (see METHODOLOGY_AUDIT.md for rationale).
  - Series construction: TTM for level/trough, YoY-delta for slope (default).
  - Within-metric combination: conjunctive (geometric mean, floors ~0.1–0.15).
  - Revenue added as sixth inflection input + cost-cut haircut.
  - Moat level: demonstrated-economics window (best 12 consecutive quarters in
    trailing ~7 yr); trailing 5-yr kept as column. **4.2 guardrail (completed
    2026-07-07 after re-audit):** short-side false-moat flags + demonstrated-
    window dates cross-referenced on the long tear sheet. Roles: the
    `false-moat-collision` tag (top-quartile demonstrated-moat rank ×
    top-quartile raw pre-haircut false-moat theme, tunable) catches
    *still-elevated* collisions; the displayed window dates catch COVID/
    cyclical spikes too old for the peak-recency primitive to fire. Display +
    tag only, no score weight; fire-rate diagnostic added (first-run item 6).
  - Missing data: exclude-and-renormalize + coverage columns; never fill.
  - Normalization: percentile ranks; sector-relative for level metrics.
  - PEG dropped for EV/Sales. Insider buying demoted to shortlist enrichment.
  - **4.7 residual clauses (completed 2026-07-07 after re-audit):**
    `on-both-lists` flag emitted when a name clears the shortlist threshold on
    both sides (independent of `contested` — neither implies the other);
    shortPercentOfFloat carried as a display-only tear-sheet column
    (frequently None for small caps), never scored — the SI denominator stays
    sharesOutstanding.
  - **Shortlist: hard cap ~100 names per side, threshold-tunable (user: surface
    significantly more than the old 20–40; expects well under the cap to
    qualify).**
  - **Soft per-sector cap: ~12–15 per sector per side, tunable (user).**
  - **Commodity discount: soft ×0.85 (tunable 0.7–1.0) on both composites for
    SIC-defined commodity price-taker sectors, tagged and visible (user —
    deliberate penalty, stronger than the audit's tag-only treatment).**
- **2026-07-08 (user direction — VITL/CAKE short archetypes; validated
  against CAKE ranking 1039/2138 pre-change):**
  - Four indicators added: run-up row (short mirror of the drawdown kernel,
    peak-evidence gated); at-peak setup primitive (level × delta-rate ×
    recency strict product, part-credits the peaked row at ×0.7 and opens
    interaction gates); pricing-masking-volume signature (GM up + revenue
    growth decaying + turnover falling); EV/normalized-EBIT valuation on
    BOTH sides (median-margin mean-reversion anchor, blended 50/50 with
    EV/Sales in the short richness row and the long moat-valuation row).
  - Inflated-growth weights rebalanced: decel .22, richness .22, run-up .18,
    pricing-masking-volume .11, DSO/DIO .09, buybacks .09, dilution .09.
  - Later same day (user decisions): **commodity discount rescinded
    entirely** (tag kept); **SI confirmation converted to an additive-only
    bonus** (max ~8 pts via the hump kernel; three-theme composite
    reweighted 0.39/0.39/0.22; absence of SI never penalizes); **exemplar
    replay harness built** (point-in-time filed-date facts + truncated
    prices; see Validation section). `archetype_ran_on_temp_success`
    discovery cohort added as a dedicated output section.
