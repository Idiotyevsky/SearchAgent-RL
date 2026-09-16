# AGENTS.md — SearchAgent-RL Research Engineering Protocol

> **Project:** SearchAgent-RL
>
> **Mission:** Reinforcement Learning for Multi-turn Search Agents.

This file governs engineering and experiment work. Public narrative belongs in
`README.md`; historical milestone records belong in `docs/archive/`.

## 1. Roles

**Sol** owns research scope, architecture, experiment design, interpretation,
resource allocation, acceptance criteria, and final claims.

**Luna Max** executes bounded, locally verifiable engineering tasks such as
parsers, loaders, metrics, tests, configuration plumbing, trajectory analysis,
plots, and approved reward formulas.

Workers must not independently redefine the benchmark, algorithm, reward
semantics, model family, or research question.

## 2. Current Research Scope

SearchAgent-RL studies how reinforcement-learning recipes change a multi-turn
search agent:

- Qwen3-8B policy;
- Hotpot-MT Strict training environment;
- Natural Bridge-Hard evaluation;
- approved MuSiQue-Local 2/3/4-hop transfer extension;
- deterministic per-trajectory BM25;
- strict one-action protocol;
- native verl/vLLM multi-turn rollouts;
- vanilla GRPO and process-aware reward comparisons;
- DAPO as an auxiliary recipe study;
- task quality, protocol reliability, and tool behavior.

Tool efficiency is one behavioral measurement, not the primary objective.
Reduced search is never considered an improvement when task quality regresses.

Out of scope unless explicitly approved:

- browser, Python, calculator, or multi-tool expansion;
- GUI or product frontend work;
- benchmark replacement;
- model-family replacement;
- distributed-RL reimplementation;
- unbounded hyperparameter search;
- multi-node scaling;
- package-wide rename.

## 3. Compatibility Invariants

The public brand is **SearchAgent-RL**, but the following identifiers remain
stable:

- Python package: `efficienttool_rl`;
- source directory: `src/efficienttool_rl/`;
- existing `ETRL_*` environment variables;
- historical run names, checkpoint names, and artifact paths;
- existing config semantics.

Do not rename these merely for branding.

Upstream verl must remain unmodified. Compatibility hooks, canonical adapters,
and reward managers belong under `src/efficienttool_rl/verl/`.

## 4. Experimental Invariants

Unless a new experiment explicitly changes one factor, preserve:

- the strict action protocol;
- `CanonicalToolAgentLoop`;
- top-1 deterministic retrieval;
- maximum five assistant turns;
- maximum three executed searches;
- 384-token observations;
- the declared dataset split and seed;
- the held-out Natural Bridge-Hard evaluator;
- complete trajectory logging.

When comparing methods, separate recipe-level conclusions from component-level
causal claims. A GRPO-vs-DAPO result does not isolate dynamic sampling,
clipping, loss aggregation, or overlong shaping unless an explicit ablation
does so.

## 5. Research Integrity

Never fabricate or extrapolate:

- EM/F1 or reward;
- training curves;
- tool counts;
- GPU throughput;
- ablation values;
- completion or validity rates.

Every public number must map to stored artifacts or a reproducible analysis
record. Label incomplete work as **in progress** or **not run**.

Do not cherry-pick one trajectory or seed as a general result. A case study must
be labeled as a case study.

Gold answers and supporting metadata may be used by reward/evaluation code but
must never enter model-visible prompts or tool kwargs.

## 6. Reward Discipline

Reward code receives special scrutiny.

For each reward change:

1. state the exact formula;
2. identify all metadata inputs;
3. prove model-visible context is unchanged;
4. test malformed and edge cases;
5. run an offline component-distribution audit;
6. log each component separately;
7. inspect group variance and reward ranking;
8. run a bounded smoke before a long experiment.

Do not change a running experiment's reward semantics. Preserve task-only
baselines and add new reward configs/adapters separately.

## 7. Diagnostics

At minimum, record:

- reward mean/std and components;
- zero-variance group ratio;
- policy loss, gradient norm, entropy, and KL where enabled;
- assistant-generated length and full trajectory length;
- attempted, valid, and executed calls;
- useful and wasted searches;
- duplicate-query and multi-search rates;
- completion, invalid action, turns, and termination reason;
- rollout and update timing.

Always inspect trajectories alongside aggregate metrics.

## 8. Failure Investigation

When a run behaves unexpectedly:

1. reproduce the symptom;
2. preserve the failing evidence;
3. verify evaluator and data split;
4. verify reward inputs and masks;
5. inspect representative trajectories;
6. verify policy parameters update;
7. inspect rollout diversity and zero-variance groups;
8. change one relevant factor;
9. rerun a bounded check;
10. record the conclusion in `docs/debug_log.md`.

Escalate when multiple root causes remain plausible, reward behavior conflicts
with manual inspection, leakage is suspected, or the next step changes scope.

## 9. GPU and Process Safety

Before a long run inspect GPU ownership/utilization, free memory, disk space,
process commands, expected checkpoint size, and output location.

Never kill a process based only on utilization. Confirm PID, user, command, and
project ownership first.

Use screen, tmux, or a scheduler so logs survive disconnects. Do not silently
overwrite output directories. Save resolved config, commit, seed, model,
dataset fingerprint, framework versions, and GPU allocation.

For OOM, capture the exact failure and identify whether it occurs in rollout or
training. Change only one obvious factor before rerunning a small smoke.

## 10. File Ownership and Parallel Work

One owner per mutable subsystem. Parallelize only independent work.

Before delegation define:

```text
Task:
Allowed files:
Do not modify:
Inputs:
Expected outputs:
Acceptance criteria:
Validation:
Return:
```

Workers must stop if completing a task requires major architecture changes,
dependency upgrades, destructive environment changes, or altered research
semantics.

## 11. Testing and Acceptance

Deterministic critical components require tests:

- parser and answer extraction;
- search environment;
- episode termination;
- task and process rewards;
- cost/usefulness accounting;
- verl adapter integration.

A success claim without test evidence is insufficient. Sol accepts a change
only after checking scope, changed files, tests, semantics, reproducibility,
leakage risk, and whether previous runs remain valid.

## 12. Documentation Boundaries

- `README.md`: stable public technical narrative and verified results.
- `experiments/`: result tables and reproducible provenance.
- `analysis/`: focused diagnostics and reward studies.
- `PROGRESS.md`: concise current status, without private machine details.
- `docs/debug_log.md`: chronological engineering audit.
- `docs/archive/`: historical milestone plans and superseded narratives.

Do not restore obsolete milestone bookkeeping to the public navigation.

## 13. Completion Standard

The project is complete only when the selected comparison has:

- a working implementation;
- stored raw trajectories and resolved configs;
- held-out task and behavior evaluation;
- reward/mask validation;
- failure analysis;
- reproducible commands;
- evidence-backed README claims;
- no fabricated results.

> Make it correct. Make it train. Make it measurable. Then interpret it.
