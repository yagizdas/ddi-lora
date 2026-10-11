# Evaluation and reported outcomes

The revised project report describes an initial comparison of the baseline and LoRA-adapted Qwen3-14B on 15 randomly selected test scenarios with different user prompts from training. A pass required identifying significant interaction pairs, explaining their mechanisms consistently with reference information, and relating the assessment to the patient context.

| Reported model | Pass | Fail | Rate |
| --- | ---: | ---: | ---: |
| Baseline | 9 | 6 | 60.0% |
| LoRA-adapted | 13 | 2 | 86.7% |

The absolute difference is 26.7 percentage points. This repository presents these as **reported historical outcomes**, not results reproduced by automated tests or an independent validation. Local experimental output files may belong to different runs and must not be silently substituted for the report's benchmark.

Table II in the paper preserves the aggregate outcomes in the original project summary. Table III measures response presentation from the recovered baseline and LoRA transcripts, which stay in the local archive. It lists word counts without assigning new case-level clinical scores. The adapted answers are shorter in 14 of 15 cases, with average length decreasing from 285.7 to 188.2 words (34.1%). Both models include the four target sections; the LoRA response for case 15 repeats them.

The original aggregate comparison and the recovered presentation measurements have different evidence sources and measure different things. Response length is not a measure of clinical correctness. With local copies of both transcripts, run `python scripts/measure_responses.py --baseline path/to/basemodel_responses.md --lora path/to/finetuned_model_responses.md` to reproduce the word counts in `response_presentation_metrics.json`.

The artifact audit measures 17 overlapping pairs out of 17 unique annotated test pairs (100%) and zero identical test/training user prompts after whitespace normalization. A post-hoc audit also identifies a drug-class error in an illustrative response. The evaluation remains small and lacks clinical pharmacologist validation. The overlap measurements and source observations are stored in `paper_artifact_audit.json`.

The current `ddi-evaluate` tool uses an LLM rubric with `pass`, `needs_review`, and `fail`. This is a separate three-way review workflow, not an implementation or reproduction of the report's original binary scoring. Its HTML interface supports inspection and human annotation. Demo verdicts are explicitly hand-authored UI fixtures.

For a new evaluation, record the train/test split, pair overlap, exact prompts, checkpoint, decoding settings, and evaluator version. Use expert review to distinguish interaction coverage, mechanism accuracy, unsupported claims, patient-context consistency, and recommendation quality. Rewritten versions of training cases should be reported as robustness checks rather than independent held-out evaluations.
