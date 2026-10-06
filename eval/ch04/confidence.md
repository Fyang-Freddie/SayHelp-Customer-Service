# ch04 confidence calibration

12 authored calibration questions; error rates are in-sample and not probability estimates.
Positive means answerable AND all gold evidence survives whole-chunk budget; missing evidence is negative.
Conservative approximate Chinese token budget: 1 character per token plus message overhead; not vendor tokenizer.
RRF rank fusion scores may tie across known/unknown questions; zero false-accept priority can reject many known questions.
Native hybrid timing includes separately labelled dense/BM25 diagnostic calls.

| strategy | threshold | positives | negatives | false accept | false reject |
|---|---:|---:|---:|---:|---:|
| dense | 0.66634983 | 8 | 4 | 0 | 0 |
| bm25 | 46.979824 | 9 | 3 | 0 | 0 |
| hybrid | 0.032786883 | 9 | 3 | 0 | 9 |
| hybrid_rerank | 2.1604438 | 9 | 3 | 0 | 0 |
