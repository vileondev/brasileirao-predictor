# CLAUDE.md — Brasileirão Scoreline & Probability Prediction Engine (v2)

## 1. Objective
Predict exact scorelines and outcome probabilities (1X2, Over/Under, BTTS) for Campeonato Brasileiro Série A matches, using only information available **before kickoff**.

**Modeling philosophy:** goals are counts. Do NOT use multi-class scoreline classification or MSE regression.
1. Estimate expected goals `λ_home`, `λ_away` (Poisson-objective GBM, or team-perspective stacked model).
2. Convert to a scoreline matrix with Dixon-Coles correction.
3. Judge everything **against baselines**, above all the bookmaker closing odds. A model that does not beat them has no value.

## 2. Data Reality (v1 scope)
Primary source: `football-data.co.uk/new/BRA.csv` — Série A 2012→present, ~380 matches/season, 14+ seasons.

| Available (v1) | NOT available (phase 2, optional) |
|---|---|
| Date, kickoff time, home, away, goals, result | xG / xGA, shots, SoT, possession, PPDA |
| Closing odds: Pinnacle (PSC*), Avg/Max (AvgC*, MaxC*) | Lineups, injuries, formations |
| (Bet365/Betfair closing: >80% missing — ignore) | Opening odds, match importance, derbies |

Consequences:
- v1 features come ONLY from results + schedule: Elo, rolling goals for/against (overall + venue), PPG form, rest days, H2H, season-progress, promoted-team flag.
- Closing odds are the **benchmark**. Using them as a feature is a separate, clearly labelled experiment (`odds_as_feature`), never the default model.
- Phase-2 sources to evaluate (verify availability/ToS first): API-Football free tier (~100 req/day, good for upcoming fixtures), Understat/FBref-like xG (Brazil coverage uncertain), CBF official data.
- Prediction moment for v1: **T-24h**, results-only. Lineup features (T-1h) only if a lineup source exists.

## 3. Anti-Leakage Rules (CRITICAL)
1. Features of match `t` use only matches strictly before kickoff of `t`. Same-day matches are ordered by kickoff timestamp; ties are NOT used for each other.
2. Never use in-match data (goals, result, shots) of the target match as input.
3. **Truncation-invariance test (mandatory):** compute features on the full dataset and on the dataset truncated right before match `t`; the feature row of `t` must be identical. Also: mutating the outcome of `t` must not change its own features.
4. Chronological splits only. Walk-forward, expanding window, with a gap (embargo) of at least one matchday between train and validation.
5. Hyperparameters (Optuna) tuned with walk-forward CV, never on the test season. The test set (latest complete season(s)) is touched once per experiment round.
6. `rho` (Dixon-Coles) is estimated on out-of-fold predictions, not on the test set.

## 4. Splits (Brasileirão)
- Warm-up: 2012–2013 (builds Elo/rolling state; not used for scoring).
- Walk-forward folds: validate on 2016, 2017, … each trained on all prior seasons.
- Final test: 2025 (complete). 2026 (in progress) = live/forward evaluation only.
- COVID season 2020 (played Aug 2020–Feb 2021 in the data labels) is kept but flagged; test sensitivity with/without it.

## 5. Repository Layout
```text
├── data/raw/                 # immutable downloads (+ download date in a sidecar .json)
├── data/interim/             # cleaned matches, canonical team ids
├── data/processed/           # feature tables
├── src/
│   ├── config.py             # paths, SEED, windows, team map, split seasons
│   ├── ingestion.py          # load, validate, normalize, dedupe
│   ├── baselines.py          # odds (de-vigged), Elo+Poisson, classic Dixon-Coles (time-decayed)
│   ├── features.py           # Elo, rolling stats, rest, H2H (strictly pre-kickoff)
│   ├── backtest.py           # walk-forward loop shared by ALL models
│   ├── train.py              # GBM Poisson / NegBin, Optuna
│   ├── probability.py        # Poisson / Dixon-Coles matrix, markets
│   ├── evaluate.py           # RPS, log-loss, Brier, Poisson deviance, calibration, bootstrap CI
│   ├── predict.py            # inference for upcoming fixtures (production Dixon-Coles)
│   ├── fixtures.py           # API-Football client (cached) + remaining-matchups helper
│   ├── forward.py            # freeze predictions before games, score afterwards
│   ├── update.py             # one-shot refresh: download, validate, refit, score, log
│   ├── stacking.py           # GBM with Dixon-Coles lambda as Poisson offset (experiment)
│   ├── stacked.py            # team-perspective stacked GBM (experiment)
│   ├── shrinkage.py          # ridge / lambda-shrinkage grid (experiment)
│   ├── dispersion.py         # Poisson vs Negative Binomial (experiment)
│   ├── calibration.py        # reliability tables and plots
│   ├── eda.py                # data-quality audit + EDA report and figures (read-only on data/raw)
│   ├── crosscheck.py         # compare scores with API-Football (seasons 2022-2024)
│   ├── eda_external.py       # audit of the external williamorim tables (stats.csv, matches.csv)
│   └── xg_api.py             # resumable API-Football match-statistics store (xG, shots, possession)
├── predictions/              # frozen forward-test logs (committed)
├── models/                   # artifacts + feature column order (git-ignored)
├── tests/                    # leakage, matrix sum=1, ingestion, inference, update, forward
├── requirements.txt
├── README.md
└── CLAUDE.md
```

## 6. Features (`src/features.py`) — v1
- **Elo** (home advantage parameter, goal-margin multiplier, carry-over regressed ~25–30% to the mean between seasons; promoted teams start at a below-mean prior). `elo_diff`.
- **Rolling goals for/against**, windows {3, 5, 10} + EWMA, overall AND venue-specific (home team's last N home games; away team's last N away games).
- **PPG form** (last 5), win/draw/loss streaks.
- **Rest days** (`rest_days_home/away/diff`; official Série A only in v1 — Copa do Brasil/Libertadores absent from source, document this limitation), matchday number, season progress.
- **H2H**: exponentially decayed goal difference, last 3–5 meetings.
- **Cold start:** fewer than N prior games → leave NaN (LightGBM handles it) plus a `games_played` feature.
- Phase 2: xG-based, shot-based, lineup/absence, formation features once data exists.

## 7. Models
1. **Baselines (build first, in `baselines.py`):** (a) de-vigged Pinnacle/Avg closing odds, (b) Elo → Poisson, (c) classic time-decayed Dixon-Coles (attack/defense per team), (d) historical league averages.
2. **GBM:** LightGBM `objective="poisson"`. Either two models (home/away) or one team-perspective stacked model (doubles rows, shares parameters — try both). Small data (~380 rows/season): strong regularization, shallow trees, few features, monotone constraints where sensible.
3. **Dispersion check:** compare Poisson vs Negative Binomial; keep only if it improves validation log-likelihood.
4. **Scoreline matrix:** grid 0..10 per side, Dixon-Coles `tau` for (0,0),(1,0),(0,1),(1,1), normalize, assert sum = 1. Extract top-3 scorelines, 1X2, O/U 2.5, BTTS.
   - `tau(0,0)=1-λHλAρ`, `tau(1,0)=1+λAρ`, `tau(0,1)=1+λHρ`, `tau(1,1)=1-ρ`.

## 8. Evaluation (`src/evaluate.py`)
- Primary: **RPS** on 1X2. Also multiclass log-loss, Brier, scoreline log-likelihood, Poisson deviance on λ.
- Always report each metric next to the **bookmaker baseline** and the Elo baseline.
- Bootstrap CIs on metric differences (paired, by match). No claims of improvement inside noise.
- Calibration curves: 1X2 and Over 2.5. Exact-score accuracy is reported but never optimized (caps ~10–13%).

## 9. Standards
- Python 3.11+ (dev machine: 3.13), `pandas`, `numpy`, `scipy`, `scikit-learn`, `lightgbm`, `optuna`, `pytest`.
- Strict type hints, NumPy-style docstrings in `src/`.
- `SEED = 42`; save feature column order with every model.
- Raw data immutable; every transformation reproducible from `data/raw`.
- Schema validation at ingestion (columns, dtypes, ranges, `Res` consistent with goals, no duplicates, no team vs itself).

## 10. Phases & Definition of Done
1. **Data:** ingestion + validation + team map. Done = tests pass, 2012–2026 loaded, no dup/NaN goals.
2. **Baselines + backtest harness:** done = RPS table (odds vs Elo vs Dixon-Coles) on walk-forward folds.
3. **Features + GBM:** done = leakage tests pass AND model compared with baselines using CIs.
4. **Calibration & probability layer:** matrix tests, `rho` fit, calibration plots.
5. **Inference:** `predict.py` for upcoming round (needs fixture source; evaluate API-Football free tier).
6. **Phase 2 data** (xG, lineups) only if a legitimate free source is verified.

## 10b. Status log
- Phase 1 (data) DONE. Phase 2 (baselines + backtest) DONE: `python -m src.backtest`.
- Baseline RPS, walk-forward 2016–2025: odds_avg 0.2005 | Dixon-Coles 0.2081 | Elo-Poisson 0.2082 | climatology 0.2188. Models trail odds by ~0.0077 (95% CI excludes 0). Test 2025: odds 0.1982 vs DC 0.1991.
- Phase 3 (features + LightGBM) DONE, run with `python -m src.train`. 53 results-only features, leakage tests pass. RESULT: GBM does NOT beat Dixon-Coles (RPS 0.2107 vs 0.2081, diff +0.0025, 95% CI [+0.0007,+0.0045]); Optuna-tuned GBM on 2023-25 hold-out 0.2113 vs DC 0.2087 (CI includes 0). GBM + closing odds as feature (EXP, labelled experiment) = 0.2046, still worse than odds alone (0.2005). Elo features dominate importance. Conclusion: ~380 rows/season is too little for a GBM on form features; next ideas: DC/Elo lambdas as offset or stacked feature, fewer features, team-perspective stacked model, pooling other leagues.
- Phase 3b (`python -m src.stacking`): GBM with Dixon-Coles lambdas as Poisson offset (14 small features / all features), out-of-sample DC lambdas for every row. RESULT: indistinguishable from DC (RPS 0.2080 vs 0.2081, diff -0.0001, CI [-0.0005,+0.0002]; hold-out 2023-25 tuned -0.0001, CI includes 0). Still +0.0074 behind odds. Conclusion: results-only features carry no extra signal beyond DC; closing odds embed information (injuries, lineups, news) we lack. Production candidate = time-decayed Dixon-Coles; further gains need NEW information, not more modeling.
- Phase 4/5 DONE (production = time-decayed Dixon-Coles): `python -m src.predict --refit --match HOME AWAY` or `--fixtures fixtures.csv` (columns home,away). Artifact: `models/dc_production.json`. Fixture sources: `--remaining` (every unplayed matchup of the double round-robin, no dates), `--fixtures` CSV; `--upcoming` (API) does not work for 2026 on the free plan.
- API-Football wired (`src/fixtures.py`, `python -m src.predict --upcoming`): key in git-ignored `.env` (API_FOOTBALL_KEY), never logged; 1 request per call, cached per day under data/raw/api_football/. VERIFIED LIVE (2026-11-24): the free plan only serves seasons 2022-2024 ("Free plans do not have access to this season"), so it cannot give 2026 fixtures; the error is cached for the day to save quota. The user's ISP DNS did not resolve api-sports.io; switching the adapter DNS to 1.1.1.1/8.8.8.8 fixed it.
- Calibration (`python -m src.calibration`, walk-forward 2016-25): 1X2 ECE 0.015 (odds 0.013), draw slightly under-predicted (25.9% vs 27.2%); Over 2.5 ECE 0.035, overconfident at high probabilities (pred 54% -> obs 47%): total-goals spread too wide, candidate fix = shrink lam_total. rho is small and window-dependent (out-of-sample on all earlier seasons: +0.03 to +0.10 in the first folds, +0.03 by 2025; last 6 seasons: -0.001; in-sample: +0.011), never the negative values of European leagues (-0.15..-0.02): no low-score draw excess. Production uses the last-6-seasons out-of-sample estimate; keep rho estimated from data, never hard-coded.
- Shrinkage experiment (`python -m src.shrinkage`): grid half-life {365,730,1460} x ridge {0.5..20} x log-lambda alpha {1..0.7}, selected on 2016-22, judged on hold-out 2023-25. RESULT: no RPS gain (selected hl=365, ridge=5, alpha=1: +0.00045, CI [-0.0007,+0.0016]; alpha=0.8: +0.00057, CI includes 0). It only improves tail calibration (home>=65%: pred .705/obs .665 -> pred .692/obs .702; over>=55%: .588/.549 -> .580/.579). Extreme matchups (Flamengo-Chapecoense ~87%) stay ~87% under stronger ridge, so they are data-driven, not under-regularization. Production DC unchanged (hl=730, ridge=0.5); adopting hl=365/ridge=5 is optional, not statistically supported.
- Forward test (`src/forward.py`): `python -m src.forward log` freezes predictions for every unplayed matchup into `predictions/<date>_remaining.csv` (immutable, commit it); `python -m src.forward score` scores them against results + bookmaker closing odds once football-data.co.uk updates `data/raw/BRA_footballdata.csv`. First log: 2026-11-24, 21 matchups (rounds 37-38, no dates). football-data.co.uk was already current to 2026-11-24. TheSportsDB free key returns only 1 upcoming event (Chapecoense x Palmeiras, 24/11 21:00, round 37) so it is not a fixture-list source; API-Football free plan is limited to seasons 2022-2024.
- Refresh command (`python -m src.update`): downloads BRA.csv, replaces the local file only if validation passes (new team names must be added to TEAM_MAP), refits and saves the production model, scores the forward test, freezes predictions for matchups not yet logged. Flags: `--no-download`, `--no-log`.
- Dispersion (`python -m src.dispersion`): Negative Binomial vs Poisson on out-of-sample DC lambdas. RESULT: the data are NOT overdispersed (E[(y-mu)^2]/E[mu] = 0.936, Pearson alpha ~ -0.05); fitted alpha hits the lower bound every season, so NegBin == Poisson (identical metrics). If anything the goals are slightly UNDER-dispersed, consistent with the absence of a low-score draw excess (rho ~ 0); a Conway-Maxwell-Poisson/generalized-Poisson could be tried but the expected gain is tiny. Poisson stays.
- Stacked team-perspective model (`python -m src.stacked`, fixed untuned params, 2016-25): one LightGBM Poisson over 2 rows/match. `stacked_plain` RPS 0.2101 (vs DC +0.0020, CI [+0.0002,+0.0038]; vs two-model GBM -0.0006, CI includes 0); `stacked_dc` (DC lambda as offset) 0.2083 (+0.0001, CI includes 0). Doubling the rows does not help: the limit is information, not sample size.
- Production rho is now out-of-sample: `predict.fit_production` fits rho on walk-forward DC lambdas of the last `RHO_SEASONS=6` seasons (-0.001) and stores `rho_in_sample` (+0.011) for reference; effect on the 21 remaining matchups <= 0.3 pp. `dc_oos_lambdas` lives in baselines.py so inference does not import lightgbm.
- Repository is PUBLIC (github.com/vileondev/brasileirao-predictor); history scanned for secrets before publishing (none). `.env` and `models/` are git-ignored. data/raw holds a copy of the football-data.co.uk CSV: check their terms before redistributing elsewhere.
- Calendar of the 21 remaining 2026 matches is hand-entered in `data/raw/schedule_2026_manual.csv` (from public schedule screenshots, local times; round 36 postponed game + rounds 37-38). All 21 inferred matchups (`fixtures.remaining_fixtures`) matched the real calendar, home/away included, which validates the double-round-robin inference. `predict --remaining` now prints dates in chronological order; tests require every remaining matchup to be in the schedule. For 2027 a new schedule file must be filled in.
- Data audit / EDA (`python -m src.eda` -> `reports/eda_report.md` + 4 figures in `reports/figures/`; spec in EDA_SPEC.md; read-only on data/raw, warning-free, tested in `tests/test_eda.py`). FINDINGS: file has 5679 rows x 32 cols (25 named + 7 empty), 15 seasons, 5678 played matches; zero duplicates, negative goals, `Res` inconsistencies or team-name drift; champion AND points match the public record for 2012-2024 (14/14 incl. 2025 champion); Pinnacle odds missing in 7.1% (MAR: depends on Season), Bet365/Betfair only partially covered (source coverage, unused). Source defects, none affecting the model: `MaxCA=0.00` in Cruzeiro x Fortaleza 27/10/2019 plus 6 atypical `Max*` prices (Max* columns are unused). 10 `Time` values have a leading space (stripped). Kickoff Time is almost certainly UTC (25% of kickoffs fall 00:00-05:59). Goals: no overdispersion (VMR HG 0.944, AG 0.999; conditional on teams 0.902/0.985); HG is slightly UNDER-dispersed (chi-square fit p=0.009), AG fits Poisson (p=0.471). Home advantage +0.45 goals/game, no significant trend (p=0.085). Year-to-year persistence of team scoring/conceding r~0.38/0.41 (supports time decay).
- Cross-source check (`python -m src.crosscheck` -> `reports/crosscheck.md`, 3 API requests, seasons 2022-2024 only on the free plan): 1140/1140 matches paired with API-Football, 0 score differences, 0 dates >1 day apart. Strong evidence the scores are real. (Needed alias `fortaleza ec` -> Fortaleza in predict.ALIASES.)
- External table `williamorim/brasileirao` `data-raw/csv/stats.csv` (MIT repo; downloaded 2026-11-24 to git-ignored `data/external/williamorim_brasileirao/`, 1,440,392 bytes, SHA-256 320283b2...fb92; underlying provenance is a Base dos Dados table, likely Transfermarkt-derived: UNVERIFIED, check terms at basedosdados.org before any redistribution). Covers seasons 2003-2020 ONLY (not 2021-2026). Columns: starting-XI market value and mean age (2012-2020: 87-100% filled), coaches, referee, attendance, standings; shots/corners/fouls/saves only 2018-2020 (44-98%). All 3,420 matches of 2012-2020 pair with ours and ALL 3,420 scores agree (second independent confirmation of the scores). Exploratory result (single pre-planned test, 1 parameter, 2016-2020, n=1,899): adding log(starting-XI value ratio) as a log-lambda shift on top of Dixon-Coles (beta ~0.03) gives RPS -0.00081 [-0.00132,-0.00030] and log-loss -0.0023 [-0.0038,-0.0007]: real but tiny (~10% of the 0.0076 gap to the odds). Cannot be used in production: no data after 2020 and no permitted source for 2021-2026. SD of the paired RPS difference between nested models is 0.0114 (vs 0.05-0.06 between very different models), see docs/PROTOCOLO.md section 4.
- External table `williamorim/brasileirao` `data-raw/csv/matches.csv` (downloaded 2026-11-24 to `data/external/williamorim_brasileirao/`, 431,150 bytes, SHA-256 840f55a5...cef7). Scraped daily (GitHub Action, 01:00 UTC) from chancedegol (2003-2019) and Globo Esporte (2020+); a maintainer must add a new scraper function each season (2027 needs `scraper_ge_2027`). 2012-2026: all 5,678 of our matches pair; scores agree in 5,675 (3 differences: 2016 Coritiba 4-0 Gremio vs 4x1, 2018 Parana 1-1 Internacional vs 1x0, 2019 Avai 0-2 Botafogo vs 0x3; the Base dos Dados stats.csv agrees with OUR scores in all three, and the user's screenshots of public result pages confirm OUR scores: Coritiba 4-0 Gremio on 07/09/2016, Parana 1-1 Internacional on 02/12/2018 (round 38), Avai 0-2 Botafogo on 04/08/2019 (round 13). So the 3 errors are in matches.csv, not in our data). It also lists all 21 remaining 2026 fixtures with dates (score `x`), and all 21 dates equal our hand-entered schedule. Date offsets vs our kickoff: 4,206 same day, 1,457 +1 day (night games, consistent with UTC), 12 outliers. Cross-source coverage of our scores: 2012-2020 (stats.csv + matches.csv), 2021-2026 (matches.csv; 2022-2024 also API-Football). Not wired into the pipeline (data/external is git-ignored; Globo Esporte terms unverified).
- External-table audit (`python -m src.eda_external` -> `reports/eda_external.md` + 4 figures `reports/figures/external_*.png`; same method as src.eda; read-only on data/external; `tests/test_eda_external.py` skips when the git-ignored files are absent). VERDICT on stats.csv: scores reliable (3,419/3,419 equal to ours, 2012-2020); dates differ by +-1 day in ~20% of matches (564 with -1, 127 with +1) and `horario` is unusable (74% of kickoffs fall 03:00-11:59; ~63% share a fixed 11 h offset to our UTC time, the rest no pattern); `chutes` cannot be total shots (`chutes_fora > chutes` in 230 of 753 team-matches); one exact duplicate row (2016-09-13 Fluminense x Atletico-MG), `id_match` not unique (2 shared ids in 2006) and 1 NaN id; 1 half-time-goals > final-goals row (2007); `faltas` up to 255 (5 rows > 60). Starting-XI value/age: 96% filled in 2012-2020 but only to 2020; nominal value median jumps 10.0 M (2014) -> 20.2 M (2015), so use the home/away RATIO (corr with goal difference 0.298 raw vs 0.299 season-normalised). Missingness is structural by season (MAR on Season); within covered seasons coach/value/age missingness vs result has p~0.01-0.02 (not significant after Bonferroni ~0.004). Shots/corners/fouls/saves only 2018-2020 (44-98%). matches.csv: 9,546 rows 2003-2026, 21 unplayed (`x`), 0 bad formats, 3 wrong scores (ours confirmed by public result pages).
- Exploratory follow-ups on the external table (2016-2020, n=1,899, hypothesis-generating, NOT confirmatory): the value-ratio gain on top of Dixon-Coles (RPS -0.0008) is uniform across the season (rounds 1-5: -0.0008, 6-15: -0.0009, 16-38: -0.0008) and for matches with or without a promoted team (-0.0007 / -0.0009), so it is NOT a cold-start fix. New-coach effect (first 5 games of a coach; tenure carried over missing names): residual goals minus lambda_dc -0.022 vs -0.005 for home teams (p=0.73) and -0.066 vs +0.022 for away teams (p=0.032, not significant after correcting for 2 tests); ~42-51 coach changes per season. Conclusion: even a clean lineup-quality signal closes only ~10% of the 0.0076 gap to the odds, so the market's edge is not lineup value (more likely injuries/suspensions of specific players, news, motivation). Nothing from this table can reach production (ends in 2020).
- API-Football xG store (`src/xg_api.py`; store in git-ignored `data/external/api_football/`; `tests/test_xg_api.py` offline). FotMob was ruled out (its ToS prohibits scraping/bulk retrieval without written consent; robots.txt blocks /api/ for generic bots). Probe 2026-11-24 (8 spread matches per season, 24 requests): xG present in 8/8 of 2023 and 8/8 of 2024, ABSENT in 8/8 of 2022 (shots/possession exist in all). Free plan: 100 requests/day, 10/min, 1 request per match => the 744 remaining matches of 2023+2024 need ~9 days at ~85/day: run `python -m src.xg_api download --budget 85 --seasons 2024 2023` once per day (resumable, never re-fetches, stops when <=10 requests are left). `python -m src.xg_api report` prints coverage. Seasons 2025-2026 are not served by the free plan, so this xG can only support research on 2023-2024 (train 2023 / test 2024 = 380 test matches), not production.
- Player-level data (squad value, absences, lineups) DEFERRED. Transfermarkt's robots.txt blocks AI crawlers (incl. ClaudeBot), so it is NOT scraped. Revisit only via a source with explicit permission (official API with key, open-licensed dataset). Compare against the baselines above if added.

## 11. Verification (run before declaring any step done)
1. Truncation-invariance + outcome-mutation leakage tests.
2. Every scoreline matrix sums to `1.0 ± 1e-6`.
3. Inference on a single mock fixture with missing optional stats works.
4. Ingestion tests on the real CSV.
