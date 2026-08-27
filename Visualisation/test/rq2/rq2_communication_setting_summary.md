# RQ2 Communication-Setting Statistics

Scope: Email vs. single-turn Vishing vs. multi-turn Vishing.

Principle values are mean WVAE persuasion scores from the six `principle_*` columns.
Pairwise differences are `right group mean - left group mean`; positive values mean the second group is larger.

## A. Data Scale

| Setting                                        |   # Samples / Conversations | Unit label                                               |   # Text Units Used | # Tokens Used   | Generator(s)          | Dataset                                                         |
|:-----------------------------------------------|----------------------------:|:---------------------------------------------------------|--------------------:|:----------------|:----------------------|:----------------------------------------------------------------|
| Email                                          |                       55095 | email samples                                            |              519304 | 9892389         | HW: 29041; LLM: 26054 | projected_points_mixed_overview.csv phishing subset             |
| Single-turn Vishing                            |                       12014 | single-turn calls/messages                               |               36195 | 281486          | HW: 10595; LLM: 1419  | HW-Vishing-single.csv; LLM-Vishing-Single.csv                   |
| Multi-turn Vishing                             |                       24111 | full conversations scored for cross-setting comparison   |              590915 | 8078696         | HW: 23905; LLM: 206   | HW-Vishing-Multi.csv; LLM-Vishing-Multi.csv                     |
| Multi-turn Vishing (round-level Fig. 2 source) |                        1458 | dialogues used for round-level malicious-turn extraction |               23838 |                 | HW: 658; LLM: 800     | HW-Vishing-Multi-ScamBaiter.csv; LLM-Vishing-Multi-BothBosu.csv |

## B. Mean Principle Scores

| Principle    |   Email |   Single-turn Vishing |   Multi-turn Vishing |   Max Difference |
|:-------------|--------:|----------------------:|---------------------:|-----------------:|
| Authority    |  0.9088 |                0.8358 |               0.9903 |           0.1545 |
| Liking       |  0.7060 |                0.5960 |               0.9378 |           0.3418 |
| Reciprocity  |  0.4134 |                0.3336 |               0.7018 |           0.3682 |
| Social Proof |  0.3139 |                0.2229 |               0.6528 |           0.4299 |
| Scarcity     |  0.0630 |                0.0241 |               0.1498 |           0.1257 |
| Commitment   |  0.0220 |                0.0076 |               0.0403 |           0.0327 |

## C1. Kruskal-Wallis Tests

| Principle    |   H statistic | p formatted   | q formatted   |
|:-------------|--------------:|:--------------|:--------------|
| Authority    |    32511.3287 | <0.001        | <0.001        |
| Liking       |    22461.4487 | <0.001        | <0.001        |
| Reciprocity  |    16953.8237 | <0.001        | <0.001        |
| Social Proof |    28063.8122 | <0.001        | <0.001        |
| Scarcity     |    29005.6690 | <0.001        | <0.001        |
| Commitment   |    23618.1082 | <0.001        | <0.001        |

## C2. Pairwise Mann-Whitney Tests

| Principle    | Comparison                                |   Difference (right-left) | q formatted   |   Effect Size (rank-biserial, right-left) |
|:-------------|:------------------------------------------|--------------------------:|:--------------|------------------------------------------:|
| Authority    | Email vs Single-turn Vishing              |                   -0.0730 | <0.001        |                                   -0.5363 |
| Authority    | Email vs Multi-turn Vishing               |                    0.0815 | <0.001        |                                    0.6285 |
| Authority    | Single-turn Vishing vs Multi-turn Vishing |                    0.1545 | <0.001        |                                    0.9499 |
| Liking       | Email vs Single-turn Vishing              |                   -0.1100 | <0.001        |                                   -0.2716 |
| Liking       | Email vs Multi-turn Vishing               |                    0.2318 | <0.001        |                                    0.5812 |
| Liking       | Single-turn Vishing vs Multi-turn Vishing |                    0.3418 | <0.001        |                                    0.7981 |
| Reciprocity  | Email vs Single-turn Vishing              |                   -0.0798 | <0.001        |                                   -0.1135 |
| Reciprocity  | Email vs Multi-turn Vishing               |                    0.2884 | <0.001        |                                    0.5270 |
| Reciprocity  | Single-turn Vishing vs Multi-turn Vishing |                    0.3682 | <0.001        |                                    0.6916 |
| Social Proof | Email vs Single-turn Vishing              |                   -0.0910 | <0.001        |                                   -0.1561 |
| Social Proof | Email vs Multi-turn Vishing               |                    0.3389 | <0.001        |                                    0.6815 |
| Social Proof | Single-turn Vishing vs Multi-turn Vishing |                    0.4299 | <0.001        |                                    0.8725 |
| Scarcity     | Email vs Single-turn Vishing              |                   -0.0389 | <0.001        |                                   -0.3961 |
| Scarcity     | Email vs Multi-turn Vishing               |                    0.0868 | <0.001        |                                    0.6360 |
| Scarcity     | Single-turn Vishing vs Multi-turn Vishing |                    0.1257 | <0.001        |                                    0.9163 |
| Commitment   | Email vs Single-turn Vishing              |                   -0.0144 | <0.001        |                                   -0.4995 |
| Commitment   | Email vs Multi-turn Vishing               |                    0.0183 | <0.001        |                                    0.4995 |
| Commitment   | Single-turn Vishing vs Multi-turn Vishing |                    0.0327 | <0.001        |                                    0.8796 |

## D. Multi-Turn Round Counts

| dataset   |   round_number |   turn_count | source_view                                                     |
|:----------|---------------:|-------------:|:----------------------------------------------------------------|
| LLM       |              1 |          424 | Fig. 2 R1-R6 cohort (dialogues with at least 6 malicious turns) |
| LLM       |              2 |          424 | Fig. 2 R1-R6 cohort (dialogues with at least 6 malicious turns) |
| LLM       |              3 |          424 | Fig. 2 R1-R6 cohort (dialogues with at least 6 malicious turns) |
| LLM       |              4 |          424 | Fig. 2 R1-R6 cohort (dialogues with at least 6 malicious turns) |
| LLM       |              5 |          424 | Fig. 2 R1-R6 cohort (dialogues with at least 6 malicious turns) |
| LLM       |              6 |          424 | Fig. 2 R1-R6 cohort (dialogues with at least 6 malicious turns) |
| HW        |              1 |          658 | all extracted malicious turns                                   |
| HW        |              2 |          653 | all extracted malicious turns                                   |
| HW        |              3 |          647 | all extracted malicious turns                                   |
| HW        |              4 |          633 | all extracted malicious turns                                   |
| HW        |              5 |          604 | all extracted malicious turns                                   |
| HW        |              6 |          581 | all extracted malicious turns                                   |
| LLM       |              1 |          800 | all extracted malicious turns                                   |
| LLM       |              2 |          799 | all extracted malicious turns                                   |
| LLM       |              3 |          786 | all extracted malicious turns                                   |
| LLM       |              4 |          688 | all extracted malicious turns                                   |
| LLM       |              5 |          535 | all extracted malicious turns                                   |
| LLM       |              6 |          424 | all extracted malicious turns                                   |
