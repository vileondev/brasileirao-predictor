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
│   └── predict.py            # inference for upcoming fixtures
├── models/                   # artifacts + feature column order
├── tests/                    # leakage, matrix sum=1, ingestion, inference
├── requirements.txt
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
- Phase 4/5 DONE (production = time-decayed Dixon-Coles): `python -m src.predict --refit --match HOME AWAY` or `--fixtures fixtures.csv` (columns home,away). Artifact: `models/dc_production.json`. Fixture source is user-supplied CSV for now (API-Football free tier not wired; check ToS first).
- API-Football wired (`src/fixtures.py`, `python -m src.predict --upcoming`): key in git-ignored `.env` (API_FOOTBALL_KEY), never logged; 1 request per call, cached per day under data/raw/api_football/. NOT yet verified live: host v3.football.api-sports.io did not resolve from the dev sandbox (DNS). Free plan may restrict seasons; confirm on first successful call.
- Calibration (`python -m src.calibration`, walk-forward 2016-25): 1X2 ECE 0.015 (odds 0.013), draw slightly under-predicted (25.9% vs 27.2%); Over 2.5 ECE 0.035, overconfident at high probabilities (pred 54% -> obs 47%): total-goals spread too wide, candidate fix = shrink lam_total. Fitted rho is POSITIVE (~+0.01 to +0.02), unlike European leagues (-0.15..-0.02): Brasileirao does not show the low-score draw excess; keep rho estimated from data, never hard-coded.
- Shrinkage experiment (`python -m src.shrinkage`): grid half-life {365,730,1460} x ridge {0.5..20} x log-lambda alpha {1..0.7}, selected on 2016-22, judged on hold-out 2023-25. RESULT: no RPS gain (selected hl=365, ridge=5, alpha=1: +0.00045, CI [-0.0007,+0.0016]; alpha=0.8: +0.00057, CI includes 0). It only improves tail calibration (home>=65%: pred .705/obs .665 -> pred .692/obs .702; over>=55%: .588/.549 -> .580/.579). Extreme matchups (Flamengo-Chapecoense ~87%) stay ~87% under stronger ridge, so they are data-driven, not under-regularization. Production DC unchanged (hl=730, ridge=0.5); adopting hl=365/ridge=5 is optional, not statistically supported.
- Forward test (`src/forward.py`): `python -m src.forward log` freezes predictions for every unplayed matchup into `predictions/<date>_remaining.csv` (immutable, commit it); `python -m src.forward score` scores them against results + bookmaker closing odds once football-data.co.uk updates `data/raw/BRA_footballdata.csv`. First log: 2026-11-24, 21 matchups (rounds 37-38, no dates). football-data.co.uk was already current to 2026-11-24. TheSportsDB free key returns only 1 upcoming event (Chapecoense x Palmeiras, 24/11 21:00, round 37) so it is not a fixture-list source; API-Football free plan is limited to seasons 2022-2024.
- Player-level data (squad value, absences, lineups) DEFERRED. Transfermarkt's robots.txt blocks AI crawlers (incl. ClaudeBot), so it is NOT scraped. Revisit only via a source with explicit permission (official API with key, open-licensed dataset). Compare against the baselines above if added.

## 11. Verification (run before declaring any step done)
1. Truncation-invariance + outcome-mutation leakage tests.
2. Every scoreline matrix sums to `1.0 ± 1e-6`.
3. Inference on a single mock fixture with missing optional stats works.
4. Ingestion tests on the real CSV.
