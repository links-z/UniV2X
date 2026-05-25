# 检测保持式协同跟踪计划书

本文档用于后续逐步执行 UniV2X 优化与论文实验。核心原则是：不推倒 UniV2X 主干，不继续把当前 direct query fusion 作为最终主方法，而是重做协同信息的作用边界，使 detection 分支稳定、tracking 分支受益。

## 0. 总目标

目标方向：

Detection-Preserved Cooperative 3D Multi-Object Tracking for V2X Perception

中文定位：

面向 V2X 协同感知的检测保持式 3D 多目标跟踪方法。

核心目标：

- mAP 尽量保持原版 UniV2X baseline，不允许再出现大幅下降。
- AMOTA 作为主要提升指标，目标超过 baseline。
- IDS 明显下降，证明协同关联有效。
- FP 不明显爆炸，避免 post-process 方法的问题。
- 当前 learnable/Sinkhorn direct fusion 保留为 ablation 和负结果分析。

建议目标区间：

| Metric | Baseline 参考 | 目标 |
| --- | ---: | ---: |
| mAP | 约 0.0526 | >= 0.050 |
| AMOTA | 约 0.1178 | >= 0.123，理想 0.130 左右 |
| IDS | 约 120 | <= 60-70 |
| FP | 约 661 | <= 720 优先 |
| Recall | 约 0.2055 | >= 0.210 |

## 1. 当前结论记录

已经观察到的问题：

- 原版 `AgentQueryFusion` 会在 detection decoder 前修改 `track_instances.query` 和 `ref_pts`。
- 原版还会把 unmatched infrastructure query 直接拼接进 ego query set。
- 因此，任何 direct fusion 都可能影响 detection 输出和 mAP。
- 当前 learnable/Sinkhorn fusion 能降低部分 IDS，但 mAP 下降过大，不适合作为最终主方法。
- post-process Sinkhorn 能拉高 AMOTA，但 FP 爆炸，论文说服力不足。

论文动机可以写成：

直接协同 query fusion 在 tracking association 上有潜力，但会扰动 detection branch，造成 mAP 不稳定。因此需要一种 detection-preserved 的协同跟踪机制。

## 2. 方法主线

最终主方法分成两个路径：

### 2.1 Detection Path

保持 ego detection path 尽量不变。

目标：

- 使用原始 ego query / ego BEV 生成 detection output。
- cooperative branch 不直接修改用于 detection 的 query、ref_pts、BEV。
- mAP 主要由 ego detection path 保证。

需要特别注意：

- 不要让 infrastructure unmatched query 直接进入 detection decoder。
- 不要让 cooperative feature fusion 改变 detection query distribution。
- 如果必须引入 cooperative 信息，也要通过 score cap、candidate confirmation 或 consistency loss 控制风险。

### 2.2 Cooperative Tracking Path

cooperative branch 只服务于 tracking association 和 track memory。

输入：

- ego track instances
- infrastructure track/query instances
- previous track memory
- ego/infrastructure score、box、velocity、feature
- calibration / pose / delay 信息

输出：

- association confidence
- updated track id / obj_idxes
- updated track memory
- candidate tracks
- calibrated tracking score

## 3. 最小可行版本 MVP

优先实现一个不复杂、可快速验证的版本。

MVP 组成：

1. Detection-preserved inference path
2. Partial-OT/Sinkhorn association
3. Candidate pool
4. Score calibration

暂时不做复杂神经网络 memory，不先追求训练很深的模块。

### 3.1 Detection-Preserved Path

在 `projects/mmdet3d_plugin/univ2x/detectors/univ2x_track.py` 中，在 cross-agent interaction 前保存 ego detection 所需变量：

- ego `track_instances`
- ego `query`
- ego `ref_pts`
- ego `bev_embed`
- ego `bev_pos`

detection output 使用保存的 ego 变量生成。

cooperative branch 的结果只用于 tracking state，不覆盖 detection path。

验收标准：

- 在不开启新 association 的情况下，mAP 应该接近原版 baseline。
- 如果 mAP 仍明显下降，说明 detection path 还没有真正隔离。

### 3.2 Partial-OT Association

新建或改造一个 association module，建议文件：

`projects/mmdet3d_plugin/univ2x/fusion_modules/coop_track_association.py`

核心功能：

- 输入 ego instances 和 infrastructure instances。
- 计算 pairwise matching cost。
- 使用 Sinkhorn / partial OT / dustbin 产生 soft matching matrix。
- 输出 matched pairs、unmatched ego、unmatched infrastructure、association confidence。

初始 cost：

```text
C = λ1 * normalized_center_distance
  + λ2 * box_size_distance
  + λ3 * velocity_distance
  - λ4 * cosine_feature_similarity
  - λ5 * score_prior
```

第一版可以先不加 uncertainty，确保基础 association 有效。

验收标准：

- IDS 比 baseline 降低。
- AMOTA 不低于 baseline。
- FP 不明显增加。

### 3.3 Candidate Pool

infrastructure unmatched query 不直接输出为正式 track。

规则：

```text
unmatched infrastructure query
  -> candidate pool
  -> 连续 K 帧确认
  -> official track
```

建议初始参数：

- `candidate_score_thr = 0.3`
- `candidate_confirm_frames = 2`
- `candidate_max_age = 3`
- `candidate_score_scale = 0.5`

验收标准：

- FP 明显低于 post-process Sinkhorn。
- Recall 不低于 baseline 太多。

### 3.4 Score Calibration

对 tracking score 做保守校准。

候选公式：

```text
final_track_score = det_score * (0.5 + 0.5 * assoc_conf)
```

更保守公式：

```text
final_track_score = min(det_score, assoc_score)
```

验收标准：

- FP 不超过 baseline 太多。
- AMOTA 相比 baseline 有提升。

## 4. 第二阶段增强创新性

MVP 成功后，再逐步增加创新模块。

### 4.1 Uncertainty-Aware Cost

为每个 observation 估计不确定性：

- detection score uncertainty
- distance/range uncertainty
- pose uncertainty
- temporal delay uncertainty
- velocity uncertainty

升级 matching cost：

```text
C = λ1 * Mahalanobis_distance
  + λ2 * velocity_distance
  + λ3 * box_size_distance
  - λ4 * feature_similarity
  - λ5 * class_score_consistency
  + λ6 * pose_uncertainty_penalty
```

论文价值：

- 可以支撑鲁棒性实验。
- 可以解释为什么在 pose noise、delay 下更稳。

### 4.2 Cooperative Track Query Memory

维护一组全局 track memory：

```text
memory_t = confirmed_tracks + candidate_tracks
```

更新逻辑：

```text
previous memory
  -> motion propagation
  -> partial OT with current observations
  -> gated memory update
```

第一版 memory update 可以用手工 gating，不必一开始就做复杂 Transformer。

后续可升级为：

- GRU-like query update
- cross-attention update
- uncertainty-gated update

论文价值：

- 将方法从单帧 fusion 提升为时序协同感知。
- 有利于降低 IDS。

### 4.3 Detection Consistency Loss

如果后续需要训练 cooperative module，可以加 detection preservation 约束。

思路：

- teacher: 原版 ego UniV2X detection output
- student: 开启 cooperative branch 后的 detection/tracking output

loss：

```text
L_preserve = L1(box_student, box_teacher)
           + KL(score_student, score_teacher)
```

注意：

- 该 loss 只用于防止 detection drift。
- 不要让它压制 tracking association 学习。

## 5. 实验计划

### 5.1 必须保留的对照

实验组：

| 编号 | 方法 | 作用 |
| --- | --- | --- |
| A | Original UniV2X | 主 baseline |
| B | Current learnable/Sinkhorn direct fusion | 证明 direct fusion 会伤 mAP |
| C | Post-process Sinkhorn | 证明 association 有潜力但 FP 高 |
| D | Detection-preserved only | 验证 mAP 可恢复 |
| E | D + Partial OT | 验证 association 提升 |
| F | E + Candidate pool | 验证 FP 控制 |
| G | F + Score calibration | 验证 AMOTA/FP 平衡 |
| H | G + Uncertainty-aware cost | 验证鲁棒性 |
| I | H + Track query memory | 最终完整方法 |

### 5.2 主要指标

必须记录：

- mAP
- AMOTA
- AMOTP
- Recall
- FP
- FN
- IDS
- MOTA 或其他 evaluator 输出的 tracking 指标

建议每个实验都整理成统一表格，不只保留日志。

### 5.3 鲁棒性实验

如果时间允许，做以下扰动：

1. pose noise
2. communication delay
3. query drop / bandwidth limit
4. infrastructure score threshold variation
5. distance range split
6. occlusion / low visibility subset

鲁棒性实验的目的：

- 证明 uncertainty-aware association 有意义。
- 证明方法不是只靠调阈值刷指标。

## 6. 消融实验写法

推荐论文表格：

### Table 1: Main Results

对比 baseline、当前方法、post-process、最终方法。

### Table 2: Ablation Study

逐步加入：

- detection-preserved
- partial OT
- candidate pool
- score calibration
- uncertainty
- memory

### Table 3: Robustness Study

在 pose noise / delay / query drop 下比较。

### Table 4: Error Analysis

重点看：

- IDS
- FP
- FN
- recall
- long-range objects

## 7. 代码执行顺序

### Step 1: 锁定 baseline

确认原版 UniV2X eval 数字，保存：

- config
- checkpoint
- eval log
- metrics summary
- git commit / diff

不需要重复跑太多次，先有一个可靠 baseline 即可。

### Step 2: 整理当前实验

把已有实验结果分组：

- baseline
- direct learnable fusion
- sinkhorn det-adapt
- post-process
- frozen-det attempt

目标：

- 不删除有用实验。
- 提取核心指标。
- 为论文 motivation 和 ablation 做准备。

### Step 3: 实现 detection-preserved switch

新增 config 开关，例如：

```python
use_detection_preserved_coop = True
```

先只验证 detection path 是否真正保持。

通过条件：

```text
mAP >= 0.050
```

如果失败，先不要继续做 association，必须先修 detection isolation。

### Step 4: 实现 association module

新增 `coop_track_association.py`。

第一版要求：

- 不训练也能跑。
- 输出 matching matrix 和 matched pairs。
- 支持 dustbin/unmatched。
- 支持阈值配置。

通过条件：

```text
IDS 下降
AMOTA 不低于 baseline
FP 不明显增加
```

### Step 5: 实现 candidate pool

把 unmatched infrastructure query 放入候选池。

通过条件：

```text
FP 相比 post-process 明显下降
AMOTA 保持或提升
```

### Step 6: 实现 score calibration

尝试两种策略：

- multiplicative score calibration
- min-score calibration

通过条件：

```text
AMOTA / FP 平衡优于 Step 5
```

### Step 7: 加 uncertainty-aware cost

先用手工 uncertainty：

- score low -> uncertainty high
- distance far -> uncertainty high
- pose noise setting -> uncertainty high

后续再考虑 learnable uncertainty head。

通过条件：

```text
clean setting 不下降
noise/delay setting 更稳
```

### Step 8: 加 track query memory

只有当前面结果稳定后再加。

优先做简单 memory：

- track state
- last feature
- age
- hit count
- miss count
- association confidence

不要一开始上复杂 Transformer memory。

通过条件：

```text
IDS 进一步下降
AMOTA 进一步提升
FP 不爆炸
```

## 8. 失败判断与回退策略

### 情况 A: mAP 还是明显下降

说明 detection path 没有隔离好。

处理：

- 检查 `bev_embed` 是否被 cooperative branch 改过。
- 检查 `ref_pts` 是否用了 fusion 后结果。
- 检查 query 数量是否变化后仍进入 detection。
- 检查 score calibration 是否影响 detection eval。

### 情况 B: AMOTA 没提升

说明 association 没带来有效收益。

处理：

- 检查 matching cost 是否过度依赖几何距离。
- 检查 association threshold 是否太严格。
- 检查 candidate pool 是否压制 recall。
- 检查 track score 是否过低。

### 情况 C: FP 爆炸

说明 candidate 或 score 太激进。

处理：

- 提高 candidate confirmation frames。
- 降低 infrastructure score scale。
- 使用 `min(det_score, assoc_score)`。
- 提高 unmatched infrastructure 接收阈值。

### 情况 D: IDS 下降但 AMOTA 不升

可能是 FP/FN 抵消了收益。

处理：

- 优先控制 FP。
- 再提高 recall。
- 不要只看 IDS。

## 9. 论文写作主线

建议论文标题：

Detection-Preserved Cooperative 3D Multi-Object Tracking for V2X Perception

摘要主线：

- V2X cooperative perception can improve tracking stability.
- Existing query-level fusion may disturb detection predictions.
- We propose a detection-preserved cooperative tracking framework.
- A partial-OT association module and candidate track pool are introduced.
- The method improves AMOTA and reduces IDS while preserving mAP.

贡献点：

1. 提出 detection-preserved cooperative tracking 框架，缓解协同 fusion 对 detection mAP 的负面影响。
2. 提出 partial-OT based cross-agent association，用于鲁棒跨智能体目标匹配。
3. 提出 candidate confirmation 和 score calibration 策略，在提升 tracking 的同时控制 FP。
4. 扩展 uncertainty-aware cost 和 track query memory，提升 pose noise / delay 场景下的鲁棒性。

## 10. 当前实验是否保留

保留，不删除。

必须保存：

- config
- train log
- eval log
- final metrics
- best/final checkpoint 至少一个
- 对应 git diff

可以删除：

- 重复中间 checkpoint
- 没有 config 和 metrics 的残缺实验
- 明显无用的临时输出

这些实验后续用于：

- motivation
- negative result
- ablation
- method comparison

## 11. 下一步立即执行

下一次开始写代码时，优先做以下三件事：

1. 整理已有实验指标表。
2. 加 `use_detection_preserved_coop` 开关。
3. 验证 detection-preserved only 的 mAP 是否回到 baseline。

只有第 3 步成功后，才继续实现新的 cooperative association。
