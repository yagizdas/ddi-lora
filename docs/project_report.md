# Patient-Aware Drug Interaction Assessment with a LoRA-Fine-Tuned Language Model

Kemal Yağız Daşkıran
Department of Electronics and Communications Engineering
Istanbul Technical University, Istanbul, Turkey
Email: daskiran21@itu.edu.tr

**Abstract—** Drug interaction databases describe medication pairs, but patient-level assessment requires us to connect those records to age, comorbidities, laboratory values, and the full medication list. We construct synthetic patient–interaction narratives from DDInter knowledge and adapt Qwen3-14B with low-rank adaptation (LoRA) to produce a four-part assessment. Our adapters update 128,450,560 parameters, approximately 0.86% of the training model. In our original evaluation of 15 randomly selected scenarios, the pass rate increased from 60.0% for the baseline to 86.7% after adaptation, an absolute improvement of 26.7 percentage points. We complement this comparison with a post-hoc error audit. Our next step is a larger, pair-disjoint evaluation with clinical expert review and retrieval before generation.

**Index Terms—** Drug–drug interactions, large language models, synthetic data, low-rank adaptation, patient context, local inference.

## I. INTRODUCTION

Medication review becomes more demanding as patients use more drugs. A regimen containing ten medications creates 45 distinct pairs. We must identify relevant interactions and then explain how the patient's age, comorbidities, laboratory values, and treatment context affect their importance. Pairwise severity labels provide a starting point for this assessment, but they do not themselves organize a complete patient-level narrative.

DDInter supplies curated interaction descriptions, severity categories, and management information [1]. These annotations provide a factual foundation for medication review. Alert fatigue also motivates careful presentation: van der Sijs et al. identified frequent overrides of computerized drug safety alerts and problems with relevance and repeated warnings [2]. We investigate whether a task-adapted language model can organize interaction knowledge into a concise assessment that keeps patient context visible.

Our research question concerns adaptation to known interactions. We train the model to identify priority pairs, explain their mechanisms, relate those risks to the patient, and present monitoring recommendations. We use these four sections as a common output structure so that readers can compare the interaction claim, supporting mechanism, contextual interpretation, and proposed management separately.

We make three contributions. First, we develop a synthetic instruction-data workflow that combines DDInter records with varied patient profiles and source-constrained target narratives. Second, we adapt Qwen3-14B through LoRA, updating approximately 0.86% of the model parameters. Third, we compare the adapted model with its unadapted backbone on 15 scenarios and audit source consistency and train/test drug-pair overlap.

We choose an open-weight backbone to support an eventual local inference workflow. In that workflow, an institution can retain control over patient inputs and model execution. Our present study focuses on dataset construction, parameter-efficient adaptation, and the quality of generated assessments.

## II. BACKGROUND AND RELATED WORK

### A. Interaction Knowledge and Patient-Level Synthesis

DDInter's original publication describes 236,834 interaction associations with mechanism and management annotations [1]. We use a subset of these records to seed instruction data. A pairwise record identifies two drugs and an interaction, whereas a patient-level assessment must also consider which risks deserve priority within the supplied profile.

Tatonetti et al. used adverse-event data to derive drug-effect and interaction signals [3]. We address a complementary task: explaining known source interactions in a patient context. The input to our generation workflow contains interaction evidence, and the output expresses that evidence in a structured narrative. This design lets us examine how adaptation changes the presentation and contextual use of an existing knowledge source.

### B. Efficient Adaptation of Open-Weight Models

LoRA adds trainable low-rank updates while freezing the pretrained backbone [4]. This approach lets us specialize the model without optimizing every weight. Qwen3-14B contains approximately 14.8 billion parameters and supports thinking and non-thinking modes [5]. We select it as an instruction-capable backbone and use its non-thinking mode for the evaluation responses.

We target both attention and feed-forward projections. Attention adapters can change how the model uses the supplied drugs and patient details, while feed-forward adapters provide additional capacity to learn the task's response pattern. Our experiment evaluates the combined adapter configuration rather than attributing its effect to a single layer group.

## III. METHODOLOGY

### A. Workflow Overview

Our workflow comprises interaction extraction, synthetic instruction generation, and supervised adaptation. We extract seed facts from DDInter, construct patient profiles and target narratives, and train Qwen3-14B on the resulting conversations. Figure 1 shows the training stages and the intended local inference path.

[FIGURE workflow]

Fig. 1. Training workflow and local inference path for a four-part patient-aware interaction assessment.

We preserve source identifiers in example metadata so that we can trace a generated narrative to its drug pair and interaction record. We also retain the source severity and mechanism category. This provenance supports manual review, later case matching, and the overlap audit in Section IV.

### B. Seed Records and Synthetic Instruction Data

We retain crawl records with a title, a known severity category, a known mechanism category, and nonempty interaction and management text. We normalize extracted whitespace and deduplicate retained records by URL, keeping the first occurrence. The saved source snapshot contains 530 DDInter records: 419 moderate interactions and 111 major interactions, or 79.1% and 20.9%, respectively.

We use GPT-5-mini to construct patient profiles and structured narratives. We instruct the generator to retain the supplied interaction and management logic, vary age and sex, incorporate relevant risk modifiers such as renal or hepatic impairment, and organize the answer into four sections. These sections cover priority interactions, mechanistic explanation, patient-specific risk assessment, and recommendations. We manually review repeated phrasing and cases that combine multiple interactions.

Our saved training snapshot contains 829 conversations: 529 source-supported interaction cases, 200 source-negative controls, and 100 multi-interaction synthesis cases. We construct source-negative controls from pairs absent from the project subset and identify them with a separate label. That label represents the evidence available in the subset; it does not assert pharmacological safety. Multi-interaction cases group several source pairs into one medication-list assessment.

We store each example as a system, user, and assistant conversation with source metadata. The user message supplies the patient profile and medications, and the assistant message supplies the target assessment. During preprocessing, we apply the backbone's chat template and shuffle the conversations with seed 3407. We use the same four-section structure across examples to make the target format consistent.

### C. LoRA Fine-Tuning

We load Qwen3-14B in 4-bit form through Unsloth and use a maximum sequence length of 2,048 tokens. We set the LoRA rank and alpha to 32, disable adapter dropout, and use seed 3407. Adapters target q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, and down_proj.

We train 128,450,560 parameters out of 14,896,757,760 total parameters. The trainable fraction is 0.86%. We use a per-device batch size of two and accumulate gradients over four steps, producing an effective batch size of eight on one device. One epoch requires 104 optimizer updates. We run the study on one NVIDIA A100, with a training duration of 28 minutes and 12 seconds.

Table I summarizes our configuration. We use a learning rate of 2 × 10⁻⁴, a linear schedule, five warmup steps, AdamW 8-bit optimization, and weight decay of 0.001. The configuration combines a quantized backbone with trainable adapters and a fixed response structure.

TABLE I. TRAINING CONFIGURATION

| Parameter | Setting |
| --- | --- |
| Base model / loading | Qwen3-14B / 4-bit |
| Maximum sequence length | 2,048 tokens |
| LoRA rank / alpha / dropout | 32 / 32 / 0 |
| Trainable parameters | 128,450,560 (0.86%) |
| Batch size / accumulation | 2 / 4 steps |
| Effective batch size | 8 |
| Epochs / optimizer updates | 1 / 104 |
| Learning rate | 2 × 10⁻⁴ |
| Scheduler / warmup | Linear / 5 steps |
| Optimizer / weight decay | AdamW 8-bit / 0.001 |
| Seed | 3407 |
| Hardware / training time | 1 × A100 / 28 min 12 s |

## IV. EXPERIMENTAL SETUP

### A. Evaluation Cases and Inference

We randomly select 15 patient scenarios from the test batch and compare the baseline and adapted backbone on the same user requests. Each request contains a patient profile and medication list. We use the educational decision-support system prompt to request source-supported reasoning, visible uncertainty, and a cautious assessment.

For inference, we disable thinking, set temperature to 0.7, top-p to 0.8, and top-k to 20, and allow up to 2,048 new tokens. We match our saved baseline and LoRA transcripts by normalized user-request text. This produces 15 paired cases in transcript order and lets us compare both responses against the same annotated source pairs.

### B. Evaluation Summary and Post-Hoc Audit

Our original evaluation uses a case-level rubric: a passing answer identifies significant interaction pairs, gives a source-consistent mechanism, and relates the assessment to the patient's context. A missed pair, an incorrect mechanism, or a contradictory contextual assessment counts as a fail. We calculate the pass rate as the number of passing scenarios divided by 15. We use the aggregate outcomes from our original project summary for Table II.

We also inspect the saved responses for repeated sections, omitted source information, and unsupported mechanisms. We use this post-hoc audit to identify improvements for the next version of the dataset and inference workflow. The audit includes checks of drug identity and class, with the illustrative methsuximide error discussed in Section V-B.

### C. Train/Test Pair-Overlap Audit

We normalize drug names by trimming whitespace and converting them to lowercase, then sort each pair so that A–B and B–A share one key. For multi-interaction cases, we include every annotated source pair. The 15 archived test cases contain 17 unique pair keys. All 17 occur in the saved training corpus, giving a measured pair overlap of 17/17, or 100%; all 15 cases include a training pair. None of the 15 user prompts exactly matches a training prompt after whitespace normalization.

## V. RESULTS AND ANALYSIS

### A. Original Evaluation Summary

Our original project summary gives nine passing baseline scenarios out of 15, or 60.0%, and 13 passing adapted-model scenarios, or 86.7%. Table II reproduces this aggregate comparison. The absolute difference is 26.7 percentage points, corresponding to four additional passes and a reduction from six failures to two. Figure 2 visualizes these counts.

TABLE II. ORIGINAL PROJECT EVALUATION SUMMARY

| Model | Pass | Fail | Pass rate |
| --- | --- | --- | --- |
| Qwen3-14B baseline | 9 | 6 | 60.0% |
| Qwen3-14B + LoRA | 13 | 2 | 86.7% |

[FIGURE outcomes]

Fig. 2. Passing and failing scenarios in the original 15-case baseline and LoRA evaluation.

### B. Qualitative Behavior and Drug-Class Audit

In cases 9 and 10, the adapted model preserves the source's direction of glycemic effect, while the baseline gives a conflicting explanation. In case 11, adaptation also produces a more source-bounded answer for a negative control. These examples illustrate how the saved models express source information within the shared response format.

A post-hoc audit identified a drug-class error in an illustrative multi-medication response: the model described methsuximide as a barbiturate. The official CELONTIN label classifies it as a succinimide anticonvulsant [6]. We use this finding to motivate explicit checks of drug identity and class, as well as interaction mechanisms.

## VI. LIMITATIONS AND FUTURE WORK

Our evaluation contains 15 synthetic scenarios, uses a project-specific rubric, and lacks clinical pharmacologist validation. The measured 100% pair overlap means that it evaluates new patient wording around familiar drug pairs rather than unseen-pair generalization. Synthetic narratives can also propagate unsupported assumptions; the source audit identifies errors in mechanism, interaction status, and multi-pair coverage. We will strengthen the scoring process through blinded expert review and separate checks of each response component.

We intend the adapted model to support local inference, but the present workflow lacks runtime retrieval and automatic source verification. We have not evaluated institutional deployment, electronic-health-record integration, regulatory compliance, or inference latency and memory.

We will expand the benchmark with pair-disjoint and mechanism-aware splits. Counterfactual patient profiles that change one relevant clinical factor will help us test whether the model responds to context rather than reproducing a familiar narrative. We will separate plausible clinical scenarios from stress tests and use blinded expert scoring to assess pair coverage, unsupported claims, mechanism accuracy, contextual consistency, and recommendation quality.

An ablation will compare full source-grounded adaptation with formatting-only targets and a prompted baseline. This comparison will help us distinguish gains from response structure, interaction knowledge, and contextual interpretation. We will also compare pair-centered training with multi-interaction examples to examine how each affects coverage in longer medication lists.

Our next implementation will enumerate medication pairs deterministically and retrieve their source records before generating the patient-level explanation. We will attach source identifiers to each interaction claim and preserve explicit uncertainty when evidence is insufficient. Expert review of the retrieved evidence and generated interpretation will guide subsequent hardware benchmarking and workflow studies.

## VII. CONCLUSION

We present a workflow for patient-aware drug interaction assessment that combines DDInter-grounded synthetic conversations with LoRA adaptation of Qwen3-14B. Our original project evaluation gives a 60.0% baseline pass rate and 86.7% with adaptation while updating approximately 0.86% of the model parameters. Our contribution includes a baseline-versus-LoRA evaluation, a post-hoc error audit, and a measured train/test pair-overlap analysis. We will next integrate retrieval before generation and evaluate the resulting assessments on a larger, pair-disjoint benchmark with clinical expert review.

## REFERENCES

[1] G. Xiong et al., “DDInter: An online drug–drug interaction database towards improving clinical decision-making and patient safety,” *Nucleic Acids Research*, vol. 50, no. D1, pp. D1200–D1207, 2022, doi: 10.1093/nar/gkab880. [Online]. Available: https://doi.org/10.1093/nar/gkab880

[2] H. van der Sijs, J. Aarts, A. Vulto, and M. Berg, “Overriding of drug safety alerts in computerized physician order entry,” *Journal of the American Medical Informatics Association*, vol. 13, no. 2, pp. 138–147, 2006, doi: 10.1197/jamia.M1809. [Online]. Available: https://doi.org/10.1197/jamia.M1809

[3] N. P. Tatonetti, P. P. Ye, R. Daneshjou, and R. B. Altman, “Data-driven prediction of drug effects and interactions,” *Science Translational Medicine*, vol. 4, no. 125, Art. no. 125ra31, 2012, doi: 10.1126/scitranslmed.3003377. [Online]. Available: https://doi.org/10.1126/scitranslmed.3003377

[4] E. J. Hu et al., “LoRA: Low-rank adaptation of large language models,” in *Proc. International Conference on Learning Representations*, 2022. [Online]. Available: https://arxiv.org/abs/2106.09685

[5] Qwen Team, “Qwen3-14B,” official model card, 2025. Accessed: Oct. 11, 2026. [Online]. Available: https://huggingface.co/Qwen/Qwen3-14B

[6] Parke-Davis, Division of Pfizer Inc., “CELONTIN (methsuximide capsules, USP),” prescribing information, revised Mar. 2023, DailyMed. Accessed: Oct. 11, 2026. [Online]. Available: https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=64a6ee88-c6b1-4e13-8208-b6772ef65a74
