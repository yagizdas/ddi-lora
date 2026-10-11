# Setup and experiment notes

## CPU tools

Install the package from the repository root with `python -m pip install -e ".[dev]"`. Core dependencies cover crawling, generation, evaluation, and review; GPU training libraries are deliberately separate. `python scripts/demo.py` uses only public fixtures and local computation.

The CLI defaults resolve relative to the current directory. Run commands from the repository root or supply explicit paths. Gemini model defaults record existing experiments; provider availability can change. Override them with `--generator-model`, `--evaluator-model`, `--rewriter-model`, or evaluator `--model` as appropriate.

## Extraction and synthetic generation

The crawler requires an explicit ID range, so invoking help or importing the module never starts a large crawl:

```bash
ddi-crawl --start-id 1 --end-id 10 --max-results 5
```

This example demonstrates the CLI; it makes network requests and does not promise that these IDs contain useful records. The output files are overwritten by crawl checkpoints, so use a separate output path for each run.

Optional source negatives are generated from missing unordered pairs within the supplied subset:

```bash
ddi-negatives --input data/raw/combined.jsonl --count 200 --include-source
ddi-generate --input data/processed/source_negatives.jsonl
```

The generator resumes based on `metadata.source_index` in an existing output. Resume only with the **same input file in the same order**; changing input records can make indices refer to different pairs. Use a new output and checkpoint for a new corpus. Generation can accept a batch after the maximum rewrite rounds, and metadata may flag validation issues; inspect generated examples before training.

## GPU notebook

`notebooks/finetune_qwen3_14b.ipynb` is a cleaned copy of the project's submitted notebook. Outputs and execution counts were removed. The inference export now writes `model_outputs`; the original version incorrectly wrote the evaluation inputs.

Use a supported CUDA GPU environment and run the notebook's installation cell. That cell preserves historical package choices, including Transformers 4.56.2 and TRL 0.22.2, while some GPU dependencies remain unpinned. It is an experiment notebook, not a fully locked environment. Hardware-specific dependencies may need adjustment, and a full GPU training run has not been performed during this cleanup.

Upload `synthetic_messages.jsonl` and `evaluation_dataset.jsonl` into the notebook's working directory. Evaluation records must contain `rewrited_messages` (produced by `ddi-rewrite`). Use independently prepared source cases for a held-out study. In Colab, uploads are separate from your local repository; locally, set the notebook working directory or adjust the file paths.

| Configuration in the notebook | Value |
| --- | --- |
| Base model | `unsloth/Qwen3-14B` |
| Loading | 4-bit |
| Maximum sequence length | 2048 |
| LoRA rank / alpha / dropout | 32 / 32 / 0 |
| Adapter modules | q/k/v/o projections; gate/up/down projections |
| Batch size / gradient accumulation | 2 / 4 |
| Epochs | 1 |
| Learning rate | 0.0002 |
| Optimizer | AdamW 8-bit |
| Warmup / scheduler | 5 steps / linear |
| Weight decay | 0.001 |
| Seed | 3407 |

These settings come from the supplied notebook. They add implementation detail beyond the PDF, but do not establish that every historical result used precisely this environment.

The notebook's inference cell generates fine-tuned answers from the in-memory trained model. Save adapters explicitly if needed with `model.save_pretrained(...)` and `tokenizer.save_pretrained(...)`. To compare a baseline, start a fresh session, load the same base model without adapters, and run identical evaluation prompts with recorded decoding settings. Trained adapters and base weights are not bundled here.

Copy exported model answers into `outputs/` and evaluate them:

```bash
ddi-evaluate --input outputs/sample_outputs_finetuned_2.jsonl
```

For an offline page using existing evaluated files:

```bash
ddi-evaluate --html-only --output examples/evaluated_outputs.jsonl --html outputs/fixture_review.html
```

Use `--comparison` together with `--non-finetuned-output` and `--fine-tuned-output` to include two additional evaluated runs.
