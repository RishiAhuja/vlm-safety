# Aggregate Statistical Checks

Rows: 1296 total, 1286 valid, 10 inference errors.
ASR is computed over valid rows only; inference errors are reported separately.

## Two-Proportion Tests

### H1 eastern-origin: West-sensitive ASR > East-sensitive ASR
- West-sensitive: 324/353 = 0.918
- East-sensitive: 273/357 = 0.765
- Difference: 0.153
- Odds ratio: 3.399
- Two-sided z-test: z=5.577, p=0.000000

### H2 western-origin: East-sensitive ASR > West-sensitive ASR
- East-sensitive: 94/180 = 0.522
- West-sensitive: 110/180 = 0.611
- Difference: -0.089
- Odds ratio: 0.697
- Two-sided z-test: z=-1.702, p=0.088804

### H3 llava_7b: eastern persona suppresses East-sensitive ASR vs no persona
- no persona: 32/60 = 0.533
- eastern persona: 28/60 = 0.467
- Difference: 0.067
- Odds ratio: 1.300
- Two-sided z-test: z=0.730, p=0.465209

### H3 minicpm_v: eastern persona suppresses East-sensitive ASR vs no persona
- no persona: 48/60 = 0.800
- eastern persona: 52/60 = 0.867
- Difference: -0.067
- Odds ratio: 0.628
- Two-sided z-test: z=-0.980, p=0.327187

### H3 qwen2.5vl_3b: eastern persona suppresses East-sensitive ASR vs no persona
- no persona: 42/57 = 0.737
- eastern persona: 39/60 = 0.650
- Difference: 0.087
- Odds ratio: 1.492
- Two-sided z-test: z=1.017, p=0.309018

## Logistic Regression

Run `python analyze_replicates.py --logit` in an environment with pandas and statsmodels
to fit `full_engagement ~ model_origin * axis + persona + category` with stimulus-clustered SEs.
The CSV tables above are the source of truth if optional dependencies are unavailable.
