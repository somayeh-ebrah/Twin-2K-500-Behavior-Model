# Evaluation Strategy

## 1. Evaluation contract

The evaluation should answer one question clearly:

> **Given a participant's leakage-safe Waves 1–3 persona, how well can the model predict that person's later behavior on the repeated Wave-4 tasks?**

For participant $i$ and target item $q$, the model receives only:

$$ X_{i,q}=(P_i,Q_q,C_{i,q},S_q) $$


where $P_i$ is the safe Waves 1–3 persona, $Q_q$ is the target question/stimulus, $C_{i,q}$ is the assigned experimental condition, and $S_q$ is the valid response schema.

The model predicts $\hat y_{i,q}$. For the primary benchmark, predictions are scored against the participant's Wave-4 response $T2$.

The same test participants also have their original response $T1$, allowing a human test–retest benchmark:


$$H_{\text{test}}=\operatorname{Score}(T1_{\text{test}},T2_{\text{test}})$$


Deliverable 1 reproduced the full-sample human test–retest score at approximately **81.73%**, but the final benchmark should recompute this number on the frozen test split only.

I treat this as a **human reliability benchmark / practical ceiling**, not a strict mathematical upper bound.

A secondary, paper-aligned score can compare the same frozen model predictions with $T1$, because the Twin-2K-500 paper reports its main digital-twin accuracy against the original held-out responses.

---

## 2. Metrics

The primary metric should remain compatible with the dataset's published evaluation logic while respecting the response type.

| Response type | Primary metric | Secondary diagnostic |
|---|---|---|
| Binary / 2-option categorical | Exact-match accuracy | Brier score / log loss if probabilities are available |
| Nominal categorical | Exact-match accuracy | Confusion matrix |
| Ordinal / Likert | Normalized absolute accuracy | Exact match, MAE |
| Bounded numeric / slider | Normalized absolute accuracy | MAE / normalized MAE |
| Anchoring free response | Decile transform, then normalized absolute accuracy | Raw MAE as a diagnostic |
| Pricing buy/not-buy | Exact-match accuracy | Brier score if probabilities are available |

For bounded ordinal or numeric responses:

$$A_{i,q}= 1- \frac{|\hat y_{i,q}-y_{i,q}|}{U_q-L_q}$$

where $L_q$ and $U_q$ are the valid minimum and maximum values. Binary questions reduce naturally to 0/1 exact match.

For anchoring questions, raw estimates are unbounded. The evaluation therefore maps them to deciles before applying the same normalized-deviation rule. In a leakage-safe benchmark, decile cut points must be fitted on the **training distribution only**, then frozen and applied to validation and test data.

### Task-level aggregation

The headline score should give equal weight to each of the 17 evaluation tasks.

For person $i$ and task $t$:

$$
A_{i,t} =
\frac{1}{|Q_{i,t}|}
\sum_{q\in Q_{i,t}} A_{i,q}
$$

The overall score is:

$$
A_{\text{macro}}
=
\frac{1}{17}
\sum_{t=1}^{17}
\frac{1}{|I_t|}
\sum_{i\in I_t} A_{i,t}
$$

Equal task weighting matters because pricing contains many more raw items than some behavioral tasks. A raw item-level mean would allow large tasks to dominate the headline result.

Every model should also report:

- accuracy for each of the 17 tasks;
- 95% confidence intervals;
- invalid-output rate;
- optional calibration metrics when the model produces probabilities.

For uncertainty, use a **participant-clustered bootstrap**: resample participant IDs, keeping all answers from a participant together.

---

## 3. Baselines and human benchmark

A behavior model should not be judged only against random guessing. The important question is whether personalization adds value beyond population-level behavior.

I would compare against the following ladder:

| Baseline | Purpose |
|---|---|
| Random legal answer | Absolute lower bound |
| Population majority / mean | Tests how far generic population behavior gets |
| Condition-specific majority / mean | Stronger trivial baseline using task and experimental condition |
| Demographics-only model | Measures signal available from simple participant attributes |
| Classical full-persona model (e.g. XGBoost) | Tests whether an LLM is needed at all |
| Prompt-only LLM | Strong language-model baseline without training |
| Proposed behavior model | Retrieval and/or fine-tuned personalized model |

The **condition-specific population baseline** is the most important trivial baseline to beat. If the model does not outperform it, apparent accuracy may come mainly from learning how people usually answer each task rather than learning the individual.

For the final test split, report:

$$
R_{\text{human}}
=
\frac{A_{\text{model}}}{H_{\text{test}}}
$$

and:

$$
G_{\text{human}}
=
H_{\text{test}}-A_{\text{model}}
$$

These give the fraction of human test–retest reliability captured by the model and the remaining gap.

---

## 4. Train / validation / test protocol

The primary split is by **participant ID**, never by answer row.

Use one frozen split manifest, for example:

```text
70% train
15% validation
15% test
```

For $N=2{,}058$, this is approximately 1,440 / 309 / 309 participants.

Commit the split once:

```text
artifacts/splits/pid_split_v1.json
```

and reuse it for every baseline, retrieval experiment, fine-tune, and ablation.

The split should be checked for reasonable balance across major demographic groups and, especially, experimental conditions.

### Strict longitudinal protocol

For the main benchmark:

1. **Train:** safe persona + T1 labels from training PIDs.
2. **Validation:** safe persona + T1 labels from validation PIDs for model selection and hyperparameter tuning.
3. **Test inference:** safe persona only for test PIDs; generate and freeze predictions.
4. **Primary test:** open T2 only after predictions are frozen and score $\hat y$ vs. T2.
5. **Human benchmark:** score T1 vs. T2 on the same test PIDs.
6. **Secondary paper-aligned score:** score the already-frozen predictions against test T1.

This keeps Wave 4 completely out of model selection and makes the primary result a genuine later-behavior test.

A secondary robustness benchmark should use **leave-one-task-out** evaluation: train on 16 tasks and evaluate the 17th. This tests whether the system learns a general persona-to-behavior relationship rather than only memorizing response patterns for known task templates.

---

## 5. Leakage prevention

### Dataset-specific trap

The most important leakage issue is the distinction between the Hugging Face persona representations.

The `full_persona` representation is **not safe for Wave-4 prediction**: for questions that occur in both earlier waves and Wave 4, it can contain information derived from the Wave-4 response. Using it as model input would leak the target.

The benchmark therefore uses only:

```text
wave1_3_persona_json
```

or summaries/retrieval indexes regenerated strictly from that safe representation.

The raw Waves 1–3 CSVs are also unsafe if used wholesale because they contain the participant's original answers to the repeated hold-out tasks.

### Additional leakage controls

| Leakage path | Prevention |
|---|---|
| `full_persona` contains Wave-4-derived information | Ban it from the primary benchmark |
| Raw Waves 1–3 include repeated-task answers | Use the released safe persona exclusion |
| Same participant in train and test | Split by PID |
| Test T1/T2 inserted into persona | Explicit target-QID exclusion tests |
| Preprocessing fitted on all participants | Fit on train only; transform val/test |
| Anchor deciles fitted globally | Fit thresholds on training data only |
| Few-shot examples from test participants | Use training PIDs only |
| Retrieval crosses participant boundaries | Enforce `retrieved_pid == target_pid` |
| Test score used for checkpoint selection | Select only on validation score |
| T1 and T2 treated as independent rows | Keep both paired under the same PID |

These rules should be enforced in code rather than only documented.

Minimum automated assertions:

```python
assert train_pids.isdisjoint(val_pids)
assert train_pids.isdisjoint(test_pids)
assert val_pids.isdisjoint(test_pids)

assert target_qids.isdisjoint(persona_qids)
assert retrieved_pid == target_pid
assert test_t2_not_loaded_during_training
assert prediction in allowed_response_schema
```

---

## 6. Acceptance criteria

A model run is accepted only if it passes both **evaluation-integrity gates** and **model-quality gates**.

### Evaluation-integrity gates

These are mandatory:

- zero PID overlap between train, validation, and test;
- zero held-out target answers in model inputs;
- no Wave-4 test labels visible before predictions are frozen;
- all preprocessing, retrieval tuning, calibration, and anchoring thresholds fitted using training/validation data only;
- at least **99.5% valid, parseable model outputs**;
- the same frozen evaluator used for every baseline and model.

Any violation invalidates the run regardless of its score.

### Model-quality gates

The candidate model should:

1. **Beat the condition-specific population baseline** on equal-task-weight test accuracy.
2. Preferably beat the strongest classical personalized baseline (e.g. XGBoost).
3. Show a positive paired-bootstrap improvement over the main trivial baseline, ideally with a 95% CI excluding zero.
4. Report performance across all 17 tasks so that the headline result is not driven by one large or easy task.

As a stronger engineering target, I would consider the model competitive if it reaches at least **90% of the test-set human reliability benchmark** while also beating the strongest non-LLM baseline. With a human benchmark near the full-sample 81.7%, that corresponds to roughly 73–74% absolute accuracy, but the threshold must be computed from the actual test-set $H_{\text{test}}$.

This 90% threshold is an engineering target for this project, not a number specified by the paper.

---

## 7. Reporting template

Every final experiment should produce one compact result table:

| Model | Macro accuracy | Human reliability ratio | Invalid outputs | Notes |
|---|---:|---:|---:|---|
| Random | ... | ... | 0% | legal-choice sampling |
| Condition population baseline | ... | ... | 0% | no persona |
| Demographics model | ... | ... | 0% | simple personalization |
| XGBoost | ... | ... | 0% | classical full-persona |
| Prompt-only LLM | ... | ... | ... | safe persona |
| Proposed LBM | ... | ... | ... | retrieval / fine-tuning |

Also include:

- a 17-task accuracy plot;
- model-minus-baseline accuracy by task;
- participant-level score distribution;
- subgroup/condition diagnostics where sample sizes are sufficient;
- a short failure analysis for the weakest tasks.

The main conclusion should answer three questions:

1. **Does the model beat a population prior?**
2. **Does personalization improve prediction for unseen people?**
3. **How much of human test–retest reliability does the model capture?**

---

## Summary

The primary benchmark is deliberately conservative: participants are split before modeling, only the leakage-safe persona is used as input, Wave 4 remains unseen until final evaluation, and performance is compared with both strong trivial baselines and the same participants' human test–retest reliability.

The most important validity condition is preventing the repeated target answers from entering the persona. The most important performance condition is beating the condition-aware population baseline. A strong model should then close a substantial fraction of the remaining gap to human self-consistency while maintaining performance across the full set of 17 behavioral tasks.
