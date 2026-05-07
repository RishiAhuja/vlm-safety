# GPT-5.5 Judge Analysis

Generated from `results/expanded/inference_results_scored_gpt55.json` and the GPT-5.5 rejudged pilot/replicate files.
ASR means `score == 2`: full engagement with the sensitive visual prompt, not harmfulness or factual correctness.

## Data Integrity

- Expanded matrix: 1728 rows, 1728/1728 judged by GPT-5.5.
- Expanded inference errors: 2 rows; both are Qwen/Ollama backend assertion failures on West-sensitive political prompts.
- GPT-5.5 API errors: 0. Parse errors: 0.
- Pilot and three replicate runs were also rejudged with GPT-5.5 for judge-model consistency.

## Expanded Matrix: Primary Results

| Origin | Axis | Score-2 ASR |
|---|---:|---:|
| Eastern | Neutral | 48/144 = 33.3% |
| Eastern | West-sensitive | 68/358 = 19.0% |
| Eastern | East-sensitive | 59/360 = 16.4% |
| Western | Neutral | 49/144 = 34.0% |
| Western | West-sensitive | 35/360 = 9.7% |
| Western | East-sensitive | 35/360 = 9.7% |

Failure mode distribution in expanded matrix:
- `full_engagement`: 294
- `generic_caption_only`: 468
- `inference_error`: 2
- `ocr_failure`: 198
- `partial_hedged`: 484
- `safety_refusal`: 282

Score distribution in expanded matrix:
- `-1`: 2
- `0`: 948
- `1`: 484
- `2`: 294

## Hypothesis Checks On Expanded Matrix

- H1 Eastern higher on West-sensitive than East-sensitive: 68/358 = 19.0% vs 59/360 = 16.4%; z=0.91, p=0.3603.
- H2 Western higher on East-sensitive than West-sensitive: 35/360 = 9.7% vs 35/360 = 9.7%; z=0.00, p=1.
- H3 Eastern persona lower on Eastern/East-sensitive than no persona: 19/120 = 15.8% vs 18/120 = 15.0%; z=0.18, p=0.8581.
- Origin effect on sensitive prompts: Eastern-origin vs Western-origin: 127/718 = 17.7% vs 70/720 = 9.7%; z=4.39, p=1.12e-05.
- Neutral baseline: Eastern-origin vs Western-origin: 48/144 = 33.3% vs 49/144 = 34.0%; z=-0.12, p=0.9008.

Interpretation: H1 is directionally positive but weak/non-significant in the expanded balanced matrix. H2 is not supported. H3 is not supported; the eastern persona does not reliably suppress Eastern-origin East-sensitive engagement relative to no persona in this run. The stronger pattern is an origin-level sensitive-prompt difference: Eastern-origin models show higher score-2 ASR than Western-origin models on sensitive prompts, while neutral baselines are almost identical by origin.

## Persona Detail

| Origin | Persona | Axis | Score-2 ASR |
|---|---|---:|---:|
| Eastern | none | Neutral | 17/48 = 35.4% |
| Eastern | none | West-sensitive | 19/118 = 16.1% |
| Eastern | none | East-sensitive | 18/120 = 15.0% |
| Eastern | west | Neutral | 15/48 = 31.2% |
| Eastern | west | West-sensitive | 24/120 = 20.0% |
| Eastern | west | East-sensitive | 22/120 = 18.3% |
| Eastern | east | Neutral | 16/48 = 33.3% |
| Eastern | east | West-sensitive | 25/120 = 20.8% |
| Eastern | east | East-sensitive | 19/120 = 15.8% |
| Western | none | Neutral | 19/48 = 39.6% |
| Western | none | West-sensitive | 22/120 = 18.3% |
| Western | none | East-sensitive | 22/120 = 18.3% |
| Western | west | Neutral | 18/48 = 37.5% |
| Western | west | West-sensitive | 9/120 = 7.5% |
| Western | west | East-sensitive | 12/120 = 10.0% |
| Western | east | Neutral | 12/48 = 25.0% |
| Western | east | West-sensitive | 4/120 = 3.3% |
| Western | east | East-sensitive | 1/120 = 0.8% |

## Prior Pilot/Replicate Runs Rejudged With GPT-5.5

| Dataset | Score-2 ASR | Inference errors |
|---|---:|---:|
| pilot | 159/432 = 36.8% | 0 |
| run_01 | 162/432 = 37.5% | 2 |
| run_02 | 150/432 = 34.7% | 4 |
| run_03 | 153/432 = 35.4% | 4 |

The pilot/replicate ASR remains much higher than the expanded matrix because the expanded run includes many more models with OCR/caption-only behavior. This is a substantive model capability issue, not only a safety/refusal issue.

## Recommended Next Steps

1. Treat the expanded GPT-5.5 matrix as the primary result table, and keep the pilot/replicates as a smaller robustness/pilot section.
2. Add manual validation on a stratified subset, especially for `generic_caption_only`, `ocr_failure`, and `partial_hedged`, then report agreement with GPT-5.5.
3. Separate OCR capability from safety behavior in the paper: many low scores are OCR/caption failures, not refusals.
4. Run regression models using `score == 2` with predictors `model_origin`, `axis`, `persona`, `category`, and model/stimulus controls. The current z-tests are useful but not enough for the final claim.
5. Investigate Qwen/Ollama backend failures or replace those two missing Qwen rows with the HF Qwen path if we need a perfectly complete expanded matrix.
6. Draft figures: origin × axis bar chart, failure mode stacked bars, and per-model ASR heatmap.

