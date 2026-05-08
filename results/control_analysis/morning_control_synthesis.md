# Morning Control Synthesis

Generated: 2026-05-08 09:31 IST after overnight PBS jobs completed.

## Completion
- PBS queue is empty; no running or held jobs remain.
- HF OCR control finished: 864/864, 0 errors.
- Ollama OCR control finished: 864/864, 0 errors.
- Ollama text-only finished and judged: 864/864 rows; 16 Qwen/Ollama backend errors.
- OCR-then-answer finished and judged: 864/864 rows; 17 Qwen/Ollama backend errors.

## Headline Numbers
- Expanded image-task, all rows: score-2 ASR 294/1726 (17.0%).
- Expanded image-task, sensitive rows only: score-2 ASR 197/1438 (13.7%).
- Ollama image-task, sensitive rows only: score-2 ASR 159/718 (22.1%).
- Ollama text-only, sensitive rows only: score-2 ASR 329/704 (46.7%).
- Ollama OCR-then-answer, sensitive rows only: score-2 ASR 171/703 (24.3%).

## OCR Diagnostics
- Ollama OCR similarity: mean=0.573, median=0.720, >=0.8: 405/864 (46.9%).
- HF OCR similarity: mean=0.640, median=0.945, >=0.8: 484/864 (56.0%).
- Granite Vision has near-zero Ollama OCR similarity and near-zero ASR; that should be framed as reading failure, not safety.
- Qwen and Gemma have near-perfect Ollama OCR, so their compliance/refusal differences are behaviorally meaningful.

## Prompt-Wrapper Effect
- Original image-task Ollama sensitive ASR: 159/718 (22.1%).
- OCR-then-answer Ollama sensitive ASR: 171/703 (24.3%).
- Text-only Ollama sensitive ASR: 329/704 (46.7%).
- The wrapper increases full engagement for some readable models, especially Gemma and Qwen, but does not rescue OCR-weak models such as Granite.
- This supports the paper angle that VLM safety evaluation must disentangle reading, captioning, and refusal.

## Backend Errors
- Remaining errors are all Qwen2.5-VL through Ollama with the same GGML assertion. Treat them as backend inference errors; we already retried text-only once.

## Tables
- `morning_summary_overall.csv`: high-level ASR/failure-mode totals.
- `morning_model_comparison.csv`: per-model image/text/OCR-then-answer comparison plus OCR quality.
