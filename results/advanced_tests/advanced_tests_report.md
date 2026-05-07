# Advanced GPT-5.5 Robustness Tests

Generated: 2026-05-08T01:12:03+05:30

## Judge Agreement: GPT-5.4-mini vs GPT-5.5

- pilot: score exact agreement=0.516, score kappa=0.245, failure-mode agreement=0.000, failure-mode kappa=0.000 (n=432).
- run_01: score exact agreement=0.560, score kappa=0.313, failure-mode agreement=0.556, failure-mode kappa=0.334 (n=432).
- run_02: score exact agreement=0.558, score kappa=0.332, failure-mode agreement=0.544, failure-mode kappa=0.335 (n=432).
- run_03: score exact agreement=0.560, score kappa=0.331, failure-mode agreement=0.551, failure-mode kappa=0.343 (n=432).
- all_pilot_replicates: score exact agreement=0.549, score kappa=0.306, failure-mode agreement=0.413, failure-mode kappa=0.225 (n=1728).

## Expanded Origin Effect Robustness
- Sensitive prompts, all models: Eastern 127/718 (17.7%), Western 70/720 (9.7%), delta=8.0 pp, p=1.12e-05.
- Two-way bootstrap over base models and stimuli: median delta=7.2 pp, 95% CI [-12.2, 30.0] pp.
- Leave-one-model-out: Eastern > Western in 11/12 removals; p<0.05 in 11/12 removals.

## Size / Edge-Model Analysis

- small_edge Neutral: 44/120 (36.7%).
- small_edge West-sensitive: 50/298 (16.8%).
- small_edge East-sensitive: 42/300 (14.0%).
- mid_large Neutral: 53/168 (31.5%).
- mid_large West-sensitive: 53/420 (12.6%).
- mid_large East-sensitive: 52/420 (12.4%).
- small_edge SensitiveCombined: 92/598 (15.4%).
- mid_large SensitiveCombined: 105/840 (12.5%).
- Small/edge vs mid/large sensitive ASR: 92/598 (15.4%) vs 105/840 (12.5%); p=0.1169.

Interpretation: the current expanded data does not justify saying compact/edge VLMs are uniformly worse. Some compact models fail badly, but size is confounded with model family and origin; per-model failure mode is the safer framing.

## Regression

See `logistic_regressions.txt`, `logistic_main_params.csv`, and `logistic_all_params.csv`. The main descriptive claim should remain origin-level sensitive-prompt ASR plus failure-mode decomposition; regression is supporting evidence, not the core figure.
