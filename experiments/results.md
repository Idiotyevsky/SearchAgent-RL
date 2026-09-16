# SearchAgent-RL Experimental Results

All completed rows below come from stored trajectory artifacts. The primary
four-policy comparison uses the same Natural Bridge-Hard set: 200 official
HotpotQA validation rows with `type=bridge`, `level=hard`, and no strict
candidate filter. A separate 1,000-example comparison checks whether the main
Vanilla GRPO versus Reward v2 conclusion persists at larger scale.

## Evaluation Comparison

| Method                           | Training reward                                   |          EM |          F1 |  Completion | Invalid action |    Searches | Multi-search | Support-hit | No-new-support |
| -------------------------------- | ------------------------------------------------- | ----------: | ----------: | ----------: | -------------: | ----------: | -----------: | ----------: | -------------: |
| Qwen3-8B Base                    | —                                                 |       32.5% |      42.03% |       93.5% |         10.06% |       1.335 |        31.5% |       0.965 |          0.370 |
| Vanilla GRPO, step 62            | task-only                                         |       51.5% |      62.53% |       97.5% |          0.17% |       1.960 |        86.0% |       1.445 |          0.515 |
| Fresh DAPO, step 62              | task-only                                         |       33.0% |      41.83% |        100% |             0% |       1.100 |        10.0% |       0.880 |          0.220 |
| GRPO + composite reward, step 62 | 0.8 / 0.15 / 0.05                                 |       45.0% |      55.25% |       98.0% |          0.52% |       1.885 |        77.5% |       1.250 |          0.635 |
| GRPO + Reward v2, step 62        | answer + annealed marginal evidence - gated waste |       45.5% |      56.83% |       99.0% |          0.37% |       1.675 |        64.0% |       1.250 |          0.425 |
| Corrected assistant-only DAPO    | task + overlong                                   | in progress | in progress | in progress |    in progress | in progress |  in progress | in progress |    in progress |

Fresh DAPO was re-aggregated from 200 rows in the stored artifact family
dapo_fresh62_nbh200/shard_{0,1,2}. Its average task reward is 0.3742, average
turns 2.10, and support-hit rate 80.0%. Its lower absolute tool cost is not an
improvement: EM regresses 18.5 percentage points relative to vanilla GRPO.

The composite-reward row was evaluated from the completed step-62 checkpoint on
all 200 rows under the same seed-42 protocol. Its average task reward is 0.5012,
average turns 2.90, average generated length 47.52 tokens, and support-hit rate
66.31%. The evaluation artifact family is
`grpo_composite_step62_nbh200_3090_20260912`.

Reward v2 completed 62 optimizer updates and was evaluated from the merged
step-62 checkpoint under the same seed-42 protocol. Its average task reward is
0.5117, average turns 2.685, average generated length 45.59 tokens, and
support-hit rate 74.63%. Search counts were 72 one-search, 121 two-search, and 7
three-search trajectories; no episode used zero searches. The training artifact
is `reward_v2_4090_4gpu_20260912_1100`, and the evaluation artifact is
`reward_v2_step62_nbh200_3090_20260913`. The evaluation data fingerprint is
`1835707b46734751610d42a6f5ebba8bb3098789f841fede1c88a63b3cbf5fdc`. Artifact
SHA-256: `trajectories.jsonl`
`92f58a6fc5fd56c0a06384c9338aa6c9a739df172ecbb0dc22f392be3e53848b`,
`metrics.json`
`2d40f2dc8364f0abbfd0936b4ada3392913fd08ee169ec8630aa43cd709e7471`, and
`run_config.json`
`b2e09d89f90a262cdb8b79dddfe5372dd6ed2321f3e6b5298a3df26a75635a2f`.

## Expanded 1,000-example Evaluation

Vanilla GRPO and Reward v2 were re-evaluated on the first 1,000 examples of the
full 5,918-example Natural Bridge-Hard pool. The original 200-example set is an
exact prefix of this larger set. Both policies used the same sampling seed and
inference settings, and the trajectory IDs align one-to-one.

Dataset SHA-256:
`46f5529eabb6a100bc9c2b0409d58ca7a99fcc6b30a27c5b6cac5a2ce7aa6234`.

| Method | EM | F1 | Completion | Invalid action | Searches | Multi-search | Support-hit | No-new-support | Support-hit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla GRPO, step 62 | **48.3%** | **62.03%** | 97.6% | 0.81% | 1.959 | **84.2%** | **1.407** | 0.552 | 71.82% |
| Reward v2, step 62 | 43.6% | 56.99% | **99.1%** | **0.23%** | **1.638** | 59.9% | 1.239 | **0.399** | **75.64%** |

Paired bootstrap differences use 20,000 resamples over the 1,000 aligned
examples and report Vanilla GRPO minus Reward v2:

| Metric | Difference | Paired bootstrap 95% CI |
| --- | ---: | ---: |
| EM | +4.70 pp | [+2.80, +6.70] pp |
| F1 | +5.03 pp | [+3.12, +7.00] pp |
| Executed searches | +0.321 | [+0.288, +0.355] |
| Support-hit searches | +0.168 | [+0.141, +0.196] |
| No-new-support searches | +0.153 | [+0.119, +0.186] |
| Multi-search | +24.3 pp | [+21.6, +27.1] pp |

The larger evaluation confirms the trade-off seen on 200 examples. Reward v2
produces a more conservative and protocol-stable policy, with fewer searches
and fewer no-new-support calls. Vanilla GRPO retains materially higher answer
quality and also retrieves more annotated supporting evidence. Reward v2 is
therefore not a Pareto improvement over the task-only objective.

Artifact SHA-256:

- Vanilla GRPO trajectories:
  `c866cd9ee02759de124488313c79bbf887801614205c74e11c4169a8151a2be4`;
- Reward v2 trajectories:
  `cc8329a01b349f23fd2c9cf056398430e179827b1d5aaa6d0213cf5a10d37de2`.

## Interpretation

- Base tends to stop after one search and under-retrieves for many bridge
  questions.
- Vanilla GRPO substantially improves task quality and moves the policy toward
  multi-step retrieval. Both support hits and no-new-support calls increase.
- Fresh DAPO is protocol-stable but one-search dominated. Its completed run did
  not apply effective overlong shaping because async prefilled scores bypassed
  that path.
- Process-aware composite GRPO improves substantially over Base but
  underperforms task-only GRPO by 6.5 EM and 7.28 F1 percentage points. It also
  produces fewer support hits and more no-new-support calls than task-only GRPO,
  so the current composite objective is not an efficiency improvement.
- Reward v2 improves over Composite v1 by 0.5 EM and 1.58 F1 percentage points,
  preserves the same support-hit count, and reduces no-new-support calls by
  0.210 per episode. It still trails task-only GRPO by 6.0 EM and 5.70 F1 points
  and retrieves less annotated supporting evidence, so it is not a Pareto
  improvement over the task-only baseline.
- The corrected assistant-only DAPO experiment is needed before drawing a causal
  conclusion about overlong shaping.

## Composite Reward Offline Audit

The process-aware reward was replayed over stored fixed-policy trajectories
before training:

| Stored policy        | Answer mean | Evidence mean | Format mean | corr(evidence, EM) | Answer share |
| -------------------- | ----------: | ------------: | ----------: | -----------------: | -----------: |
| Qwen3-8B Base        |      0.2747 |        0.5575 |      0.9483 |              0.532 |        0.627 |
| Vanilla GRPO step 62 |      0.6597 |        0.8100 |      0.9872 |              0.585 |        0.755 |

Evidence coverage carries non-zero signal; format is near ceiling and remains
auxiliary; answer reward retains the largest aggregate contribution. Raw reports
and methodology are in
[the composite reward pilot](../analysis/composite_reward_pilot/README.md).

## Metrics Required for New Rows

Each completed run must report:

- EM, token F1, completion, and invalid-action rate;
- attempted, valid, executed, support-hit, and no-new-support tool calls;
- turns, generated length, and search-count distribution;
- reward/component distributions and zero-variance group ratio;
- gradient norm, entropy, KL where enabled;
- config, seed, commit, dataset fingerprint, and artifact provenance.
