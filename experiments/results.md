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

## Zero-shot MuSiQue 2/3/4-hop Transfer

The HotpotQA-trained policies were evaluated without MuSiQue fine-tuning on a
balanced 300-example slice of the official MuSiQue-Answerable development set:
100 examples each at 2, 3, and 4 hops. Every example retains its local
20-passage candidate collection. The three policies used the same seed-42
sampling protocol and a longer interaction budget suited to the deeper chains:

- maximum 8 assistant turns;
- maximum 6 executed searches;
- deterministic top-1 BM25 retrieval;
- 384-token observations;
- 384 generated tokens per assistant turn;
- temperature 0.8 and top-p 0.95.

### Overall transfer results

| Method | EM | F1 | Task reward | Completion | Invalid action | Searches | Multi-search | Turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B Base | 9.33% | 17.19% | 0.1326 | 98.00% | 7.05% | 1.917 | 66.00% | 3.120 |
| Vanilla GRPO, step 62 | **16.33%** | **26.22%** | **0.2128** | **99.67%** | **0.09%** | 2.657 | **97.00%** | 3.660 |
| Reward v2, step 62 | 13.33% | 22.18% | 0.1776 | **99.67%** | 0.72% | 2.233 | 90.67% | 3.253 |

Vanilla GRPO transfers best overall: relative to Base it gains 7.00 EM and
9.03 F1 percentage points, increases multi-search by 31.00 points, and nearly
eliminates invalid actions. Reward v2 also improves over Base but remains below
task-only GRPO, while using 0.423 fewer searches per episode than Vanilla.

### Results by required hop count

| Required hops | Method | EM | F1 | Task reward | Searches | Multi-search | Completion | Invalid action |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | Base | 20.00% | 30.30% | 0.2515 | 1.480 | 45.00% | 97.00% | 13.12% |
| 2 | Vanilla GRPO | **37.00%** | **48.99%** | **0.4299** | 2.060 | **92.00%** | **100%** | **0.33%** |
| 2 | Reward v2 | 32.00% | 45.05% | 0.3853 | 1.800 | 78.00% | 99.00% | 2.45% |
| 3 | Base | **3.00%** | 6.12% | 0.0456 | 1.880 | 66.00% | 98.00% | 8.33% |
| 3 | Vanilla GRPO | 2.00% | **6.63%** | 0.0431 | 2.830 | **100%** | **100%** | **0%** |
| 3 | Reward v2 | 1.00% | 5.96% | 0.0348 | 2.290 | 99.00% | **100%** | **0%** |
| 4 | Base | 5.00% | 15.14% | 0.1007 | 2.390 | 87.00% | 99.00% | 0.88% |
| 4 | Vanilla GRPO | **10.00%** | **23.06%** | **0.1653** | 3.080 | **99.00%** | 99.00% | **0%** |
| 4 | Reward v2 | 7.00% | 15.53% | 0.1126 | 2.610 | 95.00% | **100%** | **0%** |

The improvement is concentrated in 2-hop questions and remains visible at
4 hops. All three policies remain weak on the 3-hop subset. Because Vanilla
uses only 2.66 searches on average despite a budget of 6, the 3-hop failure
cannot be explained by the previous three-search cap alone. Query formulation,
bridge-entity selection, and longer-horizon planning remain the main transfer
bottlenecks.

### Data and artifact provenance

- MuSiQue train artifact: 2,000 rows (800 2-hop / 700 3-hop / 500 4-hop),
  SHA-256 `0fd6f2d4fab0dcdad082d7d0bbe4c686bceeb87df241188f0ffa3450d2b4ff3f`;
- MuSiQue evaluation artifact: 300 rows (100 per hop), SHA-256
  `588ec4e05f2c1c50947e34dd122170b887496338f90d78d2a9eaf5a7a54b7354`;
- Base trajectories / metrics / resolved config:
  `3982e927ac54e996b237e053a9d0fdd8b59dbb54d2c9866a5807eff913e29ea4`,
  `531a151f773161441f41bde3ded519bfe6430980f5ed1a3658ad5d71f777429f`,
  `96fa837bd5c148b48fa5a39347bfe89fce567a1fda3a93445d0588e82a0eefe4`;
- Vanilla GRPO trajectories / metrics / resolved config:
  `f7b7039b63c323ad6e0397a3c9d007278ed2e09791c77f933e72f78ce785b6c0`,
  `f26f01334c1e8a7eca211908785588dca8e4402340f176c1f665be39b6ebd647`,
  `b6fdbf9e2d95c621de4c94414b439993e6be587e61d52bbd9ae7e0997239fb25`;
- Reward v2 trajectories / metrics / resolved config:
  `62d4032b86c87dbe884635438a181b08f6b5ea88f2ac50b47318238662bd020b`,
  `91428605d8d7aa105d4a2051492f71997cf4c399b89624d14a3c900f32632236`,
  `67c5cd8674edc90b8dfd76738c9b3c8152e0e3fd29724b36b909a3f2a53645fe`.

These results measure zero-shot policy transfer across datasets under a local
candidate-passage environment. They do not yet test MuSiQue-trained policies or
retrieval from a shared global corpus.

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
