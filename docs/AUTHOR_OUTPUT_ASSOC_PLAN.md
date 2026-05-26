# Author-Baseline Output Association Plan

本文档是从作者原版 `upstream/main` 重新开始的低风险论文方案。旧分支中的 query-level fusion 和 det-preserve 实验只作为负结果参考，不再作为主线继续堆模块。

## 结论

当前方向需要调整。

旧方案的问题不是“cooperative tracking 完全不可行”，而是实现路径风险太高：它改动了检测器内部 query/BEV 路径，导致 detection mAP 被破坏。对小论文来说，这条线的说服力和可控性都不够。

新的主线改为：

```text
Frozen Author UniV2X Detection + Output-Level Cooperative Association
```

核心原则：

- 不改作者检测网络。
- 不改 detection score、box、label 的输出口径。
- 不让任何融合模块进入 detection path。
- 只在输出层修改 `track_id`、`tracking_score` 或轨迹连续性。
- 因此 detection mAP 应该与作者 baseline 保持一致。

## 论文卖点

题目方向可以写成：

```text
Detection-Preserved Cooperative Association for V2X Multi-Object Tracking
```

核心论点：

- 端到端 query fusion 容易伤害检测精度。
- V2X 协同跟踪不一定需要改检测网络。
- 在保持检测性能的前提下，通过跨视角几何一致性、运动一致性和轻量级 OT/Hungarian 关联减少 ID switch。
- 目标是提升 AMOTA/IDS/Recall，同时保持 mAP。

## 阶段目标

### Stage 0: Clean Baseline

从作者原版代码和作者 checkpoint 跑 baseline。

验收线：

```text
mAP 与作者/已有 baseline 对齐
AMOTA、Recall、FP、IDS 作为后续对照
```

如果 clean baseline 都复现不了，先不做任何创新模块。

### Stage 1: Output-Level Association Only

实现一个后处理关联器，输入 eval 产生的预测结果。

候选匹配特征：

- 3D center distance
- class consistency
- velocity consistency
- temporal continuity
- ego/infra coordinate transform consistency

可选匹配器：

- Hungarian matching 作为主实现
- Sinkhorn/OT 作为可替换消融

验收线：

```text
mAP 不下降
IDS 下降
AMOTA 不低于 baseline，最好提升
FP 不明显增加
```

### Stage 2: Track Score Calibration

只校准 tracking score，不改 detection score。

设计：

- 短轨迹降权
- 长轨迹稳定加权
- 高速度突变降权
- 跨 agent 一致匹配加权

验收线：

```text
AMOTA 提升
IDS 不反弹
mAP 保持不变
```

### Stage 3: Candidate Pool

只在 tracking 分支增加候选池，不增加 detection 输出框。

设计：

- 保留低分但时序一致的 track candidate
- 只用于补轨迹断裂
- 不写回 detection result

验收线：

```text
Recall 或 AMOTA 提升
FP 可控
mAP 保持不变
```

## 最小消融表

只做 5 组，避免实验爆炸。

| ID | 方法 | 目的 |
| --- | --- | --- |
| A | Author UniV2X baseline | 原始基线 |
| B | Previous query fusion best/worst | 说明 query fusion 会伤 mAP |
| C | Output Hungarian association | 验证保检测关联 |
| D | C + tracking score calibration | 验证分数校准 |
| E | D + candidate pool | 最终方法 |

## 决策规则

- 如果 A 的 mAP 复现正常，继续 Stage 1。
- 如果 C 的 mAP 下降，说明实现错误，因为输出层关联不应该影响 detection mAP。
- 如果 C 只降 IDS、不升 AMOTA，继续做 D。
- 如果 D 仍无 AMOTA 提升，停止 candidate pool，论文主打 IDS/稳定性并分析 AMOTA 约束。
- 如果 E 导致 FP 大涨，回退到 D 作为最终方法。

## 为什么这个方向更稳

- 不训练也可以先出结果。
- 不依赖大规模 fine-tune。
- 失败时容易定位：因为 detection path 不动。
- 消融逻辑清楚：baseline -> association -> calibration -> candidate pool。
- 论文叙事更强：先证明 naive fusion 伤检测，再提出 detection-preserved association。

## 当前工作区

干净作者工作区：

```text
/root/autodl-tmp/UniV2X-zq/UniV2X-author-clean
```

分支：

```text
research/author-output-assoc
```

旧实验分支保留，不删除：

```text
/root/autodl-tmp/UniV2X-zq/UniV2X
research/det-preserved-coop-track
```
