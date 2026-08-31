# Long-Run Maintenance

## 1. Maintenance principle

A behavior model should not be treated as a static representation of a person. Preferences, beliefs, circumstances, products, and social context change over time, so both personas and model behavior can become stale.

The maintenance objective is:

> **Detect when the deployed system stops representing current human behavior, refresh the affected component, and keep every change auditable and reversible.**

A useful design principle is to separate **persona refresh** from **model retraining**. If one person's behavior changes, update that person's persona. Retrain the global model only when drift is systematic across many people, tasks, or contexts.

---

## 2. What to monitor

Monitor several kinds of drift because they require different responses.

| Drift | Example | Signal | Response |
|---|---|---|---|
| **Persona drift** | A person's preferences or circumstances change | new responses disagree with old persona; persona age | refresh persona |
| **Population drift** | Preferences shift across many users | feature/answer distributions move | recalibrate or retrain |
| **Task/context drift** | New products, language, or behavioral settings appear | low retrieval similarity, high uncertainty, OOD inputs | collect new human data |
| **Pipeline/model drift** | Base model, retriever, prompt, or parser changes | regression failures, invalid outputs, score changes | regression test and rollback if needed |

Operational monitoring should track:

- task-level predictive accuracy when fresh labels are available;
- response and persona-feature distributions;
- model confidence/calibration;
- invalid-output rate;
- retrieval quality and PID isolation;
- subgroup performance where sample sizes are sufficient;
- latency and inference cost.

---

## 3. Keep the human benchmark current

The original Twin-2K-500 test–retest score (~81.7%) should not be treated as a permanent lifetime ceiling. Human behavior itself can become more or less stable over time.

Where possible, maintain a small consented **refresh panel** that periodically repeats selected benchmark tasks.

Track:

$$
M_t = \text{current model accuracy}
$$

and, when repeated human responses are available,

$$
H_t = \text{current human test--retest reliability}.
$$

A useful diagnostic is:

$$
R_t = \frac{M_t}{H_t}.
$$

This helps distinguish model degradation from genuine changes in human behavioral stability.

---

## 4. Persona freshness

Every persona should have a timestamp and version.

```text
persona_v1 — Jan 2026
persona_v2 — Jun 2026
persona_v3 — Dec 2026
```

New observations should update the persona rather than silently overwrite history. Recent information can receive higher retrieval priority while older observations remain available when relevant.

Important metadata should include:

```text
persona_id
persona_version
last_updated_at
data provenance
consent / allowed purpose
```

This makes it possible to know exactly which representation of a person produced a prediction.

---

## 5. Retraining triggers

Use **scheduled evaluation plus event-driven retraining**, rather than retraining automatically on a fixed calendar.

Initial engineering triggers could include:

| Trigger | Action |
|---|---|
| Overall macro accuracy drops >3 percentage points from validated reference | investigate; retrain if persistent |
| Important task drops >5 points | task-specific investigation |
| Human-relative performance \(M/H\) declines materially | recalibrate or retrain |
| Persistent population/input drift plus behavior drift | retrain on newer data |
| Calibration deteriorates | recalibrate before full retraining |
| Invalid outputs exceed 0.5% | investigate model/pipeline regression |
| New task/domain is outside training support | collect labeled human data first |
| Base model/retriever/prompt changes materially | rerun full regression suite |

These are starting thresholds, not universal constants. In production they should be tuned to sample size, expected natural variation, and the cost of errors.

A single noisy monitoring window should not automatically trigger retraining; require persistence or statistically meaningful evidence.

---

## 6. Safe retraining process

New data should not be appended and deployed automatically.

Use a controlled **champion–challenger** process:

```text
new consented human data
        ↓
quality + leakage checks
        ↓
versioned training snapshot
        ↓
candidate model
        ↓
temporal validation
        ↓
compare with production model
        ↓
approve and deploy gradually
        ↓
monitor / rollback if needed
```

Once longitudinal data are available, use temporal backtests rather than only random splits. For example, train on earlier periods and test on later periods. This better reflects the deployment question:

> **Can past behavioral information predict future behavior?**

---

## 7. Versioning and reproducibility

A release should version the entire prediction pipeline, not only the model checkpoint.

```text
behavior-model-v3.2
├── base model
├── adapter/checkpoint
├── training-data snapshot
├── participant split
├── persona schema
├── serializer/prompt
├── retriever configuration
├── calibration parameters
└── evaluator / benchmark report
```

Every production prediction should record the relevant version IDs so it can be reproduced and audited.

If a new model, prompt, retriever, or external API version performs worse, the system should support immediate rollback to the previous validated release.

---

## 8. Governance and ethics

Long-run trustworthiness also depends on how the data and predictions are used.

### Consent and data lifecycle

Personas should only be created and refreshed from appropriately consented data. Withdrawal or deletion requests should propagate to persona stores, retrieval indexes, caches, and future training snapshots according to the applicable retention policy.

### Purpose limitation

The guardrails from the business-use design should remain enforceable after deployment. The model should not be used for:

- high-impact automated decisions about individuals;
- covert behavioral profiling;
- exploitative individualized targeting;
- personalized political manipulation;
- unsupported psychological or medical inferences.

### Subgroup monitoring

Monitor performance across sufficiently large demographic and behavioral groups. Where repeated human measurements exist, compare both model accuracy and human reliability so that natural differences in behavioral stability are not automatically labeled as model bias.

Small groups should not be reported when estimates are statistically unreliable or create privacy risk.

---

## 9. Operating cadence

A practical initial cadence is:

| Cadence | Check |
|---|---|
| Continuous | schema failures, invalid outputs, retrieval isolation, latency |
| Weekly | input/output drift and operational health |
| Monthly | task, subgroup, and calibration review when labels are available |
| Quarterly | contemporary human benchmark / temporal backtest |
| Triggered | retraining after validated drift or meaningful new data |
| Every release | full regression suite, version review, rollback readiness |

The cadence should be faster in rapidly changing domains such as pricing or consumer preferences and slower for more stable constructs.

---

## Summary

The long-run maintenance strategy separates **individual change** from **systematic model drift**.

- Refresh a persona when that person's information becomes stale.
- Recalibrate or retrain when behavior shifts across the population or tasks.
- Keep the human test–retest benchmark current rather than assuming the original ~81.7% remains fixed.
- Version the complete pipeline so every prediction is reproducible.
- Require consent, purpose controls, subgroup monitoring, and rollback capability.

The central maintenance question is not simply:

> “Is model accuracy still high?”

It is:

> **“Is the system still representing current human behavior reliably, relative to how stable that behavior itself is?”**
