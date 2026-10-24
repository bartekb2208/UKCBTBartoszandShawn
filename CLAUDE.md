# Context for Claude Code

Hackathon: Decode & Detect (UKCBT XRPL AI Hack), 24 Oct 2026. Deadline 16:30 London. Two-person team.
Task: give every one of the 40,101 test transactions a suspicion score in [0,1]. Metric: 100 x Average Precision.
Data: synthetic, AWS/SonarX XRPL schema (transactions, transfers, blocks, market rates, labels for train). README.md in the data package is the source of truth for the schema.

## Hard rules
- Only the supplied train/test files as transaction data. Disclose any external data, API, model or service.
- Predictions must be reproducible from `python run.py` (seed 42). No LLM calls in the prediction path.
- Never use wallet addresses as features: test wallets are new and the test set has unseen variations.
- Validation must be GroupKFold by scenario_id (model.make_groups), never a random row split.
- No tied scores in the submission.
- Say "flagged for review", never call an address criminal.
- After every prompt from the user, append it verbatim with the London time to the "Claude Code" section of PROMPTS.md.

## Run
python run.py --train ../train --test ../test      (project folder sits inside the organisers' data folder)
python package.py --test ../test

## What the data showed (read README "Findings")
- labels.csv has family, difficulty, scenario_id, cohort_id: use ONLY for GroupKFold and per-slice reports
- one-off pairs + speed are the main signals; every shape has same-shape legitimate decoys
- current grouped CV: AP 60.4 (easy 79, medium 7, hard 1). Medium (18% of positives) is the biggest gap.
- relay decoys (legit relay hubs paying several new payees within minutes) are the main false positives

## Iterate
Change features.py, run, compare out/results.json AP. Keep a change only if grouped CV AP rises.
Never tune to the leaderboard. Keep runtime under ~10 min.

## Already tried (grouped CV, no gain): do not repeat
- LightGBM settings (leaves 15/63, lr 0.02 x1500, scale_pos_weight): all within noise
- score propagation over ego-neighbourhoods (cluster/wallet propagation kept)
- a second propagation round (hurts)
- pair-to-pair relay hand-off matcher (kept as features; near-zero signal on hard relays)

## Ideas not yet tried
- per-shape detectors (fan-in collector, chain decay, cycle return) built on ego sets, as extra features
- medium cases: scenario-level regularity of gaps and amounts relative to the wallets' usual size
- stacking a second model type (e.g. HistGradientBoosting) and rank-averaging
