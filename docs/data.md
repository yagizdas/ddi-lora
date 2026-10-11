# Data provenance and schemas

The source of the research corpus is DDInter. The saved source snapshot contains 530 retained DDInter pairs (419 moderate and 111 major). The saved training snapshot contains 829 conversations: 529 source-supported interaction cases, 200 source-negative controls, and 100 multi-interaction cases. Original crawl files, datasets, checkpoint files, model outputs, and submission bundles remain in the ignored local archive. They are not included in the installable package or the public Git file set.

`examples/` contains three hand-authored fictional records using invented compound names and `example.org` URLs. These records exercise schemas and review tools; they are not DDInter records, clinical labels, benchmark cases, or model predictions.

## Local evaluation transcripts

The paired baseline and LoRA transcripts remain in the ignored local archive. They contain the same 15 synthetic user requests and support the paper's evaluation and post-hoc error audit. The public repository includes aggregate analysis and audit summaries; raw transcripts and generated output files stay local.

## Source records

Each JSONL line is an object with `interaction_id`, `url`, `drug_a`, `drug_b`, `severity`, `mechanism_type`, `interaction`, `management`, and optional `title`, `raw_id_field`, and `references`. The merge step deduplicates by URL, keeping the first occurrence in input order. It should run on sourced records before adding synthetic negatives, whose `url` is `N/A`.

## Conversations

Each line contains `messages`, a list of `{role, content}` objects, and `metadata` with source identifiers, pair names, severity, mechanism, and a label. The labels used by the tools are:

- `ACTIONABLE_DDI`: an actionable interaction according to the supplied source.
- `NO_SOURCE_SUPPORTED_DDI`: a pair absent from the available project source subset. This does not mean that the drugs have no interaction or are safe together.
- `MULTI_DDI_SYNTHESIS`: a separately prepared case combining several source interactions.
- `WEAK_OR_UNKNOWN_EVIDENCE`: a source record with insufficient interaction or management detail.

The rewrite tool preserves the original `messages` and writes its alternative to `rewrited_messages`. The spelling is retained for compatibility with existing experiments and the training notebook.

## Model outputs

Each line contains `index`, `metadata`, `user`, and `generated_text`. The evaluator adds an `evaluation` object with a verdict, confidence, summary, issues, and human review priority. LLM verdicts are review aids; they are not expert clinical labels.

## Preparing your data

Place authorized source exports in `data/raw/`. For existing local materials, inspect `local_archive/term_project_files/` and copy only the files you intend to use into the new paths. Copy operations preserve the archive. File counts and contents differ across the historical experiment snapshots, so avoid treating arbitrary filenames as a canonical split.

Use separate training and evaluation files, document their provenance, and check overlap at the pair level. Adding source-negative examples or multi-interaction cases creates a new experiment and should be recorded separately from the reported 830-example run.

The public repository does not redistribute the full DDInter-derived corpus. Public accessibility of a database does not by itself determine redistribution rights. Check the source's current usage terms before crawling or distributing data. The crawler respects robots checks and stops on blocked or rate-limited responses; it has not been exercised against the live website during repository preparation.
