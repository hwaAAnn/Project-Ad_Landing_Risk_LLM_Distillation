# LLM Distillation for Ad-Landing Risk Classification

An experimental framework that compares the risk judgments of three commercial LLMs (Claude, Gemini, OpenAI) on ad landing pages under **identical inputs and identical criteria**, then uses the resulting gold labels to **distill** the judgment into a 4B-parameter open model (Qwen3-4B, QLoRA).

---

## Abstract

The task is to classify the landing page of a mobile ad as risky or not. We treat commercial LLMs as teachers and fine-tune a small open model (the student, Qwen3-4B with QLoRA) on labels derived from their judgments. To make this measurable, we (1) freeze every judgment input as a sealed snapshot, removing confounders from cross-model comparison; (2) concentrate human labeling effort on the cases where models disagree; and (3) score agreement-derived and human-resolved subsets separately, so that imitation of the teachers can be told apart from real discriminative power.

QLoRA fine-tuning lifts Qwen3-4B from 75.8% to 85.9% accuracy (Macro-F1 0.742 → 0.846), above two of the three commercial teachers, with the highest RISK precision (91.0%). The strongest teacher, Claude Haiku 4.5, remains ahead at 92.0% (see Results).

## Results

| Model | Type | Accuracy | RISK Recall | RISK Precision | Macro-F1 |
| :-- | :-- | --: | --: | --: | --: |
| Gemini 2.5 Flash | Commercial LLM | 84.1% | 86.7% | 89.5% | 0.824 |
| **Claude Haiku 4.5** | Commercial LLM | **92.0%** | **96.7%** | 87.9% | **0.921** |
| GPT-5 mini | Commercial LLM | 81.6% | 84.3% | 87.6% | 0.801 |
| Qwen3-4B | Zero-shot | 75.8% | 72.4% | 78.2% | 0.742 |
| Qwen3-4B | QLoRA fine-tuned | 85.9% | 87.8% | **91.0%** | 0.846 |

Binary task. The production grading is three-way (`LOW` safe · `MEDIUM` caution · `HIGH` danger). For this comparison, caution and danger are merged into one positive class: RISK = `MEDIUM · HIGH`, SAFE = `LOW` (rationale in §2.5). Figures are as reported from the project's experiment runs. Raw predictions and adapter weights are not released (§3.3). The pipeline in §4.5 regenerates the table, and `distill/eval_binary.py` also reports danger misses (gold `HIGH` predicted SAFE), the costliest error for a protection service.

## 1. Motivation

LLM-based judges are accurate but costly per call, add latency, and depend on external APIs. This task has structured inputs (a bundle of tool-call results) and a small output space (an ordinal 3-grade label plus a type), which makes distillation into a lightweight classifier a realistic option. Distillation presupposes that **teacher quality can be measured reliably**, and that in turn requires the comparison itself to be controlled.

## 2. Experimental Design

### 2.1 Pipeline

```
dataset/cases.jsonl        (curated set: includes raw tool_outputs)
dataset/prompt_spec.json   (judging criteria: system prompt + output schema, sealed by hash)
        │
        ├─ build_prompt.py   raw tool_outputs → user prompt, assembled with the production rules
        │
        ├─ run_eval.py --provider claude  →  results/claude/{run_id}.jsonl
        ├─ run_eval.py --provider gemini  →  results/gemini/{run_id}.jsonl
        └─ run_eval.py --provider openai  →  results/openai/{run_id}.jsonl
                                                   │
                  merge_results.py  →  results/merged/compare_{tag}.jsonl, .csv
                                                   │
                  extract_disagreements.py  →  dataset/to_label_{tag}.json
                                                   │   (annotator fills in `expect` for disagreements)
                  build_gold.py     →  gold/gold_251.jsonl
                                                   │
                  score_gold.py     →  results/scored/errors_{tag}.csv
                                                   │
                  build_trainset.py →  dataset/trainset_{tag}.jsonl   (student training set)
                                                   │
                  finetune_qlora.py →  adapters/qwen3-4b-qlora/       (QLoRA adapter)
                  predict_local.py  →  predictions/{name}.jsonl       (zero-shot and fine-tuned)
                                                   │
                  eval_binary.py    →  results table (teachers and students, same rows)
```

### 2.2 Controlled Input

The following are held fixed so that the comparison is fair.

- **Frozen snapshots.** Tool results (`tool_outputs`) collected at judgment time are stored and are the only thing read. No page is re-opened, no ad is re-clicked, and no blocklist or RDAP lookup is repeated, so repeated runs see identical input.
- **Sealed judging criteria.** The system prompt and output schema are pinned in `prompt_spec.json`, and `system_sha256` is recomputed on every run. A mismatch aborts the run, so model differences are never conflated with criteria differences, whenever the run happens.
- **Assembly conditions (`--input-mode`).** `production-trim` (the production length limits), `raw` (no trimming) and `stored` (use a stored prompt) isolate the effect of input truncation as a separate experiment.
- **Leakage control.** `lookup_cache`, which holds earlier model judgments, is excluded from the input by default. It is the one path by which a previous teacher's output could flow into the input of the model under evaluation. Tools removed from the input are also removed from the reference text used for evidence grounding (`_ground`).
- **Sampling parameters.** Production uses API defaults, so `temperature` is left unset. It is overridden with `--temperature 0` only for determinism experiments, and the value is recorded in each result row.
- **Output schema enforcement.** All three models are held to the same schema, each through its native mechanism. The conversion logic lives in one place, `pipeline/providers/base.py`.

| provider | schema enforcement | default model |
| :-- | :-- | :-- |
| claude | `tools=[final_verdict]` + forced `tool_choice` | `claude-haiku-4-5` |
| gemini | `response_schema` + JSON mime type | `gemini-2.5-pro` |
| openai | `response_format: json_schema` (strict) | `gpt-5` |

### 2.3 Gold Label Construction

Inter-model agreement measures how similar the models are to each other, not how accurate they are. If all three models are wrong together, agreement is still 100%, so accuracy can only be defined against separate gold labels.

1. Cases on which all three models agree are **auto-confirmed** with the consensus label (`source: auto_agree`).
2. Only cases with at least one disagreement are extracted by `extract_disagreements.py` and **resolved by a human**.
3. `build_gold.py` merges both into a gold set of 251 cases.

This concentrates labeling cost on the disagreement region, at the price that the agreement region inherits the teachers' bias (see §5).

### 2.4 Evaluation Protocol

`score_gold.py` reports accuracy on three subsets.

| subset | meaning |
| :-- | :-- |
| All | all 251 cases |
| 3-way agreement | the 220 auto-confirmed cases. All three models score 100% by definition, and **if they do not, the scorer is wrong and the run aborts** (sanity check) |
| Human-resolved | the resolved disagreements. **Real discriminative power only shows up here** |

Grades are ordinal (`LOW < MEDIUM < HIGH`), so accuracy is reported together with a split into under-calls and over-calls. Runs that failed to satisfy the schema are recorded rather than dropped, and the schema-compliance failure rate is itself a comparison metric.

### 2.5 Distillation Dataset

`build_trainset.py` turns the gold labels into a binary training problem for the student.

- **Label.** `LOW = 0`, `MEDIUM · HIGH = 1`. There is only one HIGH case, so three-class training and evaluation are not viable and the labels are binarized.
- **Input.** The user prompt assembled by `build_prompt.py`.
- **Re-derivability.** The original `risk · type · source` are kept on every row, so when more data arrives the label definition can change without re-assembling inputs.
- **Input identity check.** The assembled prompts are compared against the prompts actually sent to the teachers, for every case (`prompt_sha256` in `results/*/*.jsonl`). A single mismatch aborts the build, because student and teacher scores cannot share a table if their inputs differ. Assembly parameters are read from a reference run rather than hardcoded.
- **Splits.** Rows are stratified by `label × source` into a held-out test split (20%) and, independently, into 5 CV folds, so the human-resolved cases are spread across both.

### 2.6 Student Model

| item | design |
| :-- | :-- |
| Base model | `Qwen/Qwen3-4B`, thinking mode off |
| Adaptation | QLoRA: 4-bit quantized base with trainable LoRA adapters |
| Prompt | the sealed judging criteria + one-word answer instruction (`SAFE` / `RISK`), then the same user prompt the teachers received |
| Loss | answer tokens only. The prompt is masked, so the model learns the mapping, not the page text |
| Inference | P(RISK) from the next-token logits of the two label words. No free-form generation, so no parse failures |
| Long inputs | the head of the user text is kept and the tail dropped. The criteria are never truncated |

Zero-shot and fine-tuned runs share the prompt format (`local_judge.py`), so the adapter is the only difference between the two Qwen rows.

## 3. Reproducibility

### 3.1 Results Record

Every result is one JSONL line per case, flushed per case. No database or server is involved, so each experimenter produces only their own files and merges them at the end.

```json
{"case_id":"case_001","run_id":"20260829T1000Z-gemini-50dcb7",
 "provider":"gemini","model":"gemini-2.5-pro",
 "spec_id":"oneshot-20260828","system_sha256":"50dcb7…",
 "input_mode":"production-trim","excluded_tools":["lookup_cache"],"temperature":null,
 "prompt_sha256":"a1b2…","ok":true,
 "verdict_raw":{"risk":"HIGH","type":"impersonation","reason":"…","advice":"…","evidence":"…"},
 "verdict":    {"risk":"MEDIUM","type":"impersonation","reason":"…","advice":"…","evidence":"…"},
 "postprocess":{"fill_type_applied":false,"ground_downgraded":true},
 "latency_ms":1840,"usage":{"input_tokens":5120,"output_tokens":172},
 "attempts":1,"error":null}
```

- `verdict_raw` is the raw model output and `verdict` is the value after post-processing (`_fill_type` → `_ground`). Both are kept so that "model alone" and "model inside the system" evaluations can be chosen after the fact.
- `ground_downgraded: null` means there was no `tool_outputs` to check against, so the downgrade decision was **deferred**. It distinguishes a downgrade caused by missing evidence from a deferral caused by missing data.
- `run_id` is `{UTC time}-{provider}-{first 6 chars of the criteria hash}`, so a re-run never overwrites earlier results.
- `input_mode`, `temperature` and `prompt_sha256` are recorded on every row, so it can be verified afterwards that results from different conditions were not mixed into one merge. Experiments with changed conditions must use a separate `--out-dir`.

### 3.2 Pinned Artifacts

| item | value |
| :-- | :-- |
| `spec_id` | `oneshot-20260828` |
| `system_sha256` | `50dcb7dcef47475353d87837a0a6ab2f42c017b9bf043bd02f032732dec3cc6e` |
| `prompt_spec.json` SHA256 | `af5c136ed645603f52bd08d2ceb6c92bcf328d11b016baceceb9d2c087234033` |
| `cases.jsonl` SHA256 | `2e6aea4b491f33c758d7254628c292aee67f2bb5c5b468482bfd8a6d64963392` |
| `cases.jsonl` size | 100 lines, 711,954 bytes |

### 3.3 Data Availability

The raw dataset (`cases.jsonl`) and model judgment results (`results/*`) are **not released**. Page bodies contain personal data and third-party copyrighted material. What is released:

| item | released | note |
| :-- | :-: | :-- |
| Judging criteria (`prompt_spec.json`) | ✅ | sealed by hash |
| Gold labels (`gold/gold_251.jsonl`) | ✅ | only `case_id · url · risk · type` |
| Selection conditions and id mapping | ✅ | provenance record, no page bodies |
| Synthetic examples (`cases.example.jsonl`) | ✅ | 3 cases, for the input format |
| Raw cases, model judgments, training set | ❌ | contain page text and judgment sentences |
| Student predictions and LoRA adapters | ❌ | derived from the unreleased training set |

### 3.4 Input Format

One line is one case. The raw data is stored and **assembly happens at run time**. To measure how much the truncation policy changes judgments with `--input-mode raw`, the originals have to be preserved.

| field | required | description |
| :-- | :-: | :-- |
| `case_id` | ✅ | merge key. Duplicates abort at load time |
| `input.url` | ✅ | landing address reached by the ad click |
| `input.tool_outputs` | ✅ | raw tool returns at judgment time: `[{"tool","args","output"}, …]` |
| `input.click_url` | | the clicked ad link. Added to the prompt only when it differs from `url` |
| `reference` | | an earlier judgment. A **comparison baseline**, not ground truth |

The order of `tool_outputs` carries no meaning. In production the tools run in parallel and the order differs every time, so the assembler identifies slots by `(tool name, arguments)`.

### 3.5 Assembly Fidelity

Input assembly and post-processing (`build_prompt.py`, `postprocess.py`, `domains.py`) reproduce production code, so they must be re-synced by hand when the original changes. `test_assembly.py` checks fidelity in three ways.

| check | source of ground truth | needs original repo |
| :-- | :-- | :-: |
| A | `prompt_text` from real collected pages | ✅ |
| B | output of the original `_prefetch` run against a fake `run_tool` | ✅ |
| C | `input_mode` behavior and `lookup_cache` exclusion | ❌ |

Ground truth is extracted from the original code rather than written by hand, so if the original assembly logic changes, this check fails first.

The two optional environment variables below point at the original service and are only needed for regenerating the criteria and for checks A and B. Judging runs never need them.

| optional feature | environment variable |
| :-- | :-- |
| Regenerate criteria: `build_prompt.py --from-server` | `JUDGE_SERVER_SRC` |
| Assembly checks A and B in `test_assembly.py` | `JUDGE_SERVICE_ROOT` |

## 4. Usage

### 4.1 Setup

```bash
pip install -r requirements.txt     # installing only the SDKs you need is fine
```

Every command runs from the repository root as a module (`python -m pipeline.run_eval`, `python -m tests.test_score_gold`).

API keys are passed by environment variable only and are never written to code or result files.

| provider | environment variable |
| :-- | :-- |
| claude | `ANTHROPIC_API_KEY` |
| gemini | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| openai | `OPENAI_API_KEY` |

### 4.2 Dry run

Check input quality before spending money, with no API key or SDK.

```bash
python -m pipeline.run_eval --dry-run
python -m pipeline.run_eval --dry-run --input-mode raw
```

It checks that cases load, that prompts build, how many inputs are truncated, the prompt length distribution (min/median/p90/max), `tool_outputs` parse failures, and slot-recovery problems (missing core slots, duplicates, unknown tools). Any finding gives exit code 1. Full per-case prompts are saved to `results/dryrun/`. `--dry-run` blocks every outbound connection.

### 4.3 Evaluation

```bash
python -m pipeline.run_eval --provider claude --limit 3            # smoke test
python -m pipeline.run_eval --provider gemini                      # default conditions
python -m pipeline.run_eval --provider claude --resume             # continue from the interruption point

# ablations
python -m pipeline.run_eval --provider claude --input-mode raw --out-dir results/claude-raw
python -m pipeline.run_eval --provider claude --temperature 0     --out-dir results/claude-t0

python -m pipeline.merge_results --tag 0829                        # merge + agreement + pairwise disagreements
python -m pipeline.merge_results --tag 0829 --raw                  # before post-processing (raw model output)
```

While running, `run_eval.py` blocks network connections to anything other than the provider API hosts (`--no-guard` disables this, which is not recommended).

### 4.4 Scoring and Training Set

```bash
python -m pipeline.score_gold results/merged/compare_0830.jsonl
python -m tests.test_score_gold              # verifies scoring logic (no network)

python -m pipeline.build_trainset --tag 0831
python -m tests.test_build_trainset          # verifies extraction logic (no network)
```

### 4.5 Model Comparison and Distillation

Teachers. The table uses the smaller tier of each vendor, selected with `--model`. Keep each in its own `--out-dir`.

```bash
python -m pipeline.run_eval --provider gemini --model gemini-2.5-flash --out-dir results/gemini-flash
python -m pipeline.run_eval --provider claude --model claude-haiku-4-5 --out-dir results/claude
python -m pipeline.run_eval --provider openai --model gpt-5-mini      --out-dir results/openai-mini
```

Student. Needs a CUDA GPU and `pip install -r requirements-local.txt`.

```bash
T=dataset/trainset_0831.jsonl

python -m distill.predict_local  --trainset $T --name qwen3-4b-zeroshot            # zero-shot
python -m distill.finetune_qlora --trainset $T                                     # → adapters/qwen3-4b-qlora
python -m distill.predict_local  --trainset $T --name qwen3-4b-qlora --adapter adapters/qwen3-4b-qlora
```

Scoring. Every model is scored on the same held-out rows of the training set.

```bash
python -m distill.eval_binary --trainset $T \
    --run  "Gemini 2.5 Flash=results/gemini-flash/<run_id>.jsonl" \
    --run  "Claude Haiku 4.5=results/claude/<run_id>.jsonl" \
    --run  "GPT-5 mini=results/openai-mini/<run_id>.jsonl" \
    --pred "Qwen3-4B zero-shot=predictions/qwen3-4b-zeroshot.jsonl" \
    --pred "Qwen3-4B QLoRA=predictions/qwen3-4b-qlora.jsonl"

python -m tests.test_eval_binary            # verifies metrics and row selection (no network, no GPU)
```

Cross-validation instead of a single split: pass `--cv-fold k` (0–4) to `finetune_qlora.py`, `predict_local.py` and `eval_binary.py` alike.

## 5. Limitations and Threats to Validity

- **Teacher-imitation bias.** 220 of the 251 gold cases were auto-confirmed by 3-way agreement. A student trained on this data largely learns to imitate the teachers, so results must always be reported per subset using `source`.
- **Class imbalance.** There is only one HIGH case, so three-class classification and HIGH recall cannot be measured with the current data. Binarization is a stopgap.
- **Sample size.** The human-resolved subset is too small to claim statistical significance for differences between models.
- **Annotator variability.** Human resolution of disagreements relies on a single rubric, and inter-annotator agreement was not measured.
- **Distribution shift.** The curated set was filtered by collection time, de-duplication and input completeness, so it does not represent the full production distribution.
- **Post-processing coupling.** `verdict` is the model output combined with rule-based post-processing. Model-only performance should be read from `verdict_raw`.
- **Small evaluation set.** The held-out split is a fraction of 251 cases, so a difference of one or two cases moves accuracy by several points. The gap between the fine-tuned student and the weaker teachers is within that range. Cross-validation (`--cv-fold`) gives a steadier estimate.
- **Student ceiling.** Because most labels come from teacher agreement, the student is bounded by the teachers on that subset. Its advantage in precision should be confirmed on the human-resolved cases specifically.

## 6. Repository Layout

```
.
├── pipeline/                   teacher comparison → gold labels → training set
│   ├── providers/              base.py (shared contract, schema conversion) · claude · gemini · openai
│   ├── build_prompt.py         criteria seal check, raw tool_outputs → user prompt
│   ├── schema.py               case and result schemas (single source)
│   ├── postprocess.py          _fill_type · _ground post-processing (reproduction)
│   ├── domains.py              host / site helpers for slot identification
│   ├── run_eval.py             judge all cases with one provider, --dry-run
│   ├── merge_results.py        merge by case_id, agreement, pairwise disagreements
│   ├── extract_disagreements.py
│   ├── extract_references.py
│   ├── build_gold.py           disagreement resolution → gold labels
│   ├── score_gold.py           per-subset scoring, under-/over-call split
│   └── build_trainset.py       gold + assembled prompts → student training set
├── distill/                    student model
│   ├── local_judge.py          prompt format, loading, label-logit scoring (shared)
│   ├── finetune_qlora.py       QLoRA fine-tuning of Qwen3-4B
│   ├── predict_local.py        zero-shot / fine-tuned inference
│   └── eval_binary.py          binary metrics for teachers and students on the same rows
├── tests/                      offline checks (no network, no GPU)
├── dataset/                    judging criteria, id mapping, synthetic examples
├── gold/                       gold_251.jsonl
├── results/                    run outputs (not committed)
├── requirements.txt            provider SDKs
└── requirements-local.txt      GPU stack for the student
```

## Note on language

Prompt text assembled for the models (section headers and fallback strings such as the placeholder for a tool that did not return in time) and most code comments are in Korean, because the judged content is Korean. The prompt text is part of the pinned input, so it is intentionally left as is.
