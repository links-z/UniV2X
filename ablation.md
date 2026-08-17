# UniV2X 改进方法三项消融实验计划书

## 1. 实验目的

本消融实验旨在验证本文所提出的 Learnable Sinkhorn-based Query Fusion 方法中三个关键组件的有效性，分别为：

1. Sinkhorn soft matching；
2. confidence-aware gate；
3. new-object complement。

通过分别去除上述模块，并与完整模型进行对比，分析各模块对车路协同多目标跟踪性能的具体贡献。实验重点关注 AMOTA、MOTA、Recall、FP、IDS 和 FAF 等指标，以验证本文方法是否能够提升跨智能体查询匹配质量、抑制错误融合，并增强路侧目标补充能力。

---

## 2. 实验基本原则

为保证消融实验公平性，所有消融实验均采用统一设置：

```text
共同初始化：Stage2-300 checkpoint
训练阶段：Stage3
训练长度：300 iterations
优化器设置：与 Ours Full 的 Stage3 设置一致
数据集：nuScenes validation set
评估协议：与 UniV2X 原始评估协议一致
```

除被消融的目标模块外，其余网络结构、学习率、训练数据、batch size、评估脚本和 checkpoint 保存间隔均保持一致。

本消融实验不追求每个变体单独调参后的最高性能，而是通过统一训练条件比较各模块的相对贡献。

---

## 3. 共同起点设置

所有消融实验均从 Stage2-300 checkpoint 开始。

选择 Stage2-300 作为统一起点的原因是：此前实验发现，冻结 bbox head 训练到 600 iterations 后结果反而下降，说明过长的冻结检测头训练可能导致跨智能体匹配分布发生漂移。因此，Stage2-300 更适合作为后续 Stage3 微调和消融对比的共同初始化点。

配置中建议设置为：

```python
load_from = 'projects/work_dirs_e2e_univ2x/xxx_stage2_300iter.pth'
resume_from = None
auto_resume = False
```

其中 `xxx_stage2_300iter.pth` 替换为实际的 Stage2-300 权重路径。

---

## 4. 统一 Stage3 微调设置

所有消融实验均采用相同的 Stage3 设置：

```python
optimizer = dict(
    type='AdamW',
    lr=5e-5,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.0),
            'img_neck': dict(lr_mult=0.0),
            'cross_agent_query_interaction': dict(lr_mult=0.05),
            'pts_bbox_head': dict(lr_mult=0.02),
        }
    )
)
```

训练长度统一为：

```text
Stage3-300 iterations
```

checkpoint 保存建议：

```python
checkpoint_config = dict(
    by_epoch=False,
    interval=100,
    max_keep_ckpts=10,
    filename_tmpl='ablation_iter_{}.pth'
)
```

每个实验建议保存：

```text
iter_100.pth
iter_200.pth
iter_300.pth
```

最终主表统一比较 `iter_300.pth` 的评估结果。

---

## 5. 消融实验一：w/o Sinkhorn

### 5.1 实验名称

```text
Ours w/o Sinkhorn
```

### 5.2 实验设置

去除 Sinkhorn soft matching 模块，不再使用带 dustbin 的 Sinkhorn 归一化匹配矩阵。

完整方法中：

```text
S = S_app + S_geo + S_conf
S → Sinkhorn normalization → matching matrix M
```

消融后：

```text
不使用 Sinkhorn soft matching
```

可替换方式包括：

```text
1. 使用原始 UniV2X 的 cross-agent query interaction；
2. 或使用简单相似度直接融合；
3. 或使用 top-1 / hard matching 作为简化匹配。
```

需要保证其余模块保持不变，即：

```text
Gate：保留
Complement：保留
bbox fine-tune：保留
```

### 5.3 实验目的

验证 Sinkhorn soft matching 是否能够有效建立 vehicle queries 与 infrastructure queries 之间的全局软关联关系。

### 5.4 重点观察指标

```text
AMOTA
MOTA
IDS
FP
FAF
```

### 5.5 预期现象

如果 Sinkhorn soft matching 有效，去除该模块后，跨智能体查询匹配会变得不稳定，可能导致：

```text
AMOTA下降
MOTA下降
IDS上升
FP或FAF上升
```

### 5.6 论文可用解释

去除 Sinkhorn 软匹配后，模型无法显式建模车端查询与路侧查询之间的全局匹配关系，跨智能体目标关联稳定性下降，从而影响整体跟踪性能。

---

## 6. 消融实验二：w/o Gate

### 6.1 实验名称

```text
Ours w/o Gate
```

### 6.2 实验设置

保留 Sinkhorn soft matching，但去除 confidence-aware gate。

完整方法中：

```python
fused_feat = veh_feat + gate * delta
```

其中 `gate` 由 Sinkhorn matching confidence 或 match_conf 生成，用于控制路侧融合特征注入车端 query 的强度。

消融后：

```python
fused_feat = veh_feat + delta
```

或者等价设置为：

```python
gate = 1.0
```

需要保证其余模块保持不变，即：

```text
Sinkhorn：保留
Complement：保留
bbox fine-tune：保留
```

### 6.3 实验目的

验证 confidence-aware gate 是否能够抑制低置信度匹配带来的错误融合。

### 6.4 重点观察指标

```text
AMOTA
FP
IDS
FAF
Recall
```

### 6.5 预期现象

如果 confidence-aware gate 有效，去除该模块后，所有匹配特征会被无条件注入车端查询，低质量匹配和错误路侧信息更容易进入融合结果，可能导致：

```text
FP上升
IDS上升
FAF上升
AMOTA下降
```

Recall 可能保持不变或略有上升，但如果错误融合过多，整体 AMOTA 仍可能下降。

### 6.6 论文可用解释

去除置信度感知门控后，模型无法根据匹配可靠性动态控制路侧特征的融合强度，低质量匹配可能引入噪声，从而增加误检或身份切换。

---

## 7. 消融实验三：w/o Complement

### 7.1 实验名称

```text
Ours w/o Complement
```

### 7.2 实验设置

保留 Sinkhorn soft matching 和 confidence-aware gate，但关闭 new-object complement 机制。

完整方法中：

```text
unmatched infrastructure queries → new-object complement → enhanced vehicle queries
```

消融后：

```text
不补充未匹配的 infrastructure queries
```

需要保证其余模块保持不变，即：

```text
Sinkhorn：保留
Gate：保留
bbox fine-tune：保留
```

### 7.3 实验目的

验证 new-object complement 是否能够利用路侧感知信息补充车端漏检目标，从而提升目标覆盖能力。

### 7.4 重点观察指标

```text
Recall
FN
AMOTA
MOTA
```

### 7.5 预期现象

如果 new-object complement 有效，去除该模块后，未匹配的路侧目标无法被引入车端查询集合，可能导致：

```text
Recall下降
FN上升
AMOTA下降
```

IDS 可能变化不明显，因为该模块主要影响目标覆盖能力，而不是直接控制身份关联。

### 7.6 论文可用解释

去除新目标补充机制后，路侧端观测到但车端未充分感知的目标无法被补充到车端查询集合中，导致目标召回能力下降。

---

## 8. 完整方法对照：Ours Full

虽然本计划主要包含三个消融实验，但实验表中必须加入完整方法作为对照。

### 8.1 实验名称

```text
Ours Full
```

### 8.2 实验设置

完整保留以下模块：

```text
Sinkhorn soft matching
+ confidence-aware gate
+ new-object complement
+ Stage3 bbox head fine-tuning
```

### 8.3 作用

作为三个消融实验的参照对象，用于证明完整模块组合能够取得最优或较优的综合跟踪性能。

如果已有 Ours Full Stage3-300 结果，可以直接使用已有结果，无需重复训练；但必须保证其起点、训练长度和学习率设置与三个消融实验一致。

---

## 9. 实验运行目录建议

建议为三个消融实验分别建立独立目录：

```text
projects/work_dirs_e2e_univ2x/ablation_wo_sinkhorn_stage3_300
projects/work_dirs_e2e_univ2x/ablation_wo_gate_stage3_300
projects/work_dirs_e2e_univ2x/ablation_wo_complement_stage3_300
```

完整方法目录：

```text
projects/work_dirs_e2e_univ2x/ablation_full_stage3_300
```

checkpoint 命名建议：

```text
wo_sinkhorn_iter_100.pth
wo_sinkhorn_iter_200.pth
wo_sinkhorn_iter_300.pth

wo_gate_iter_100.pth
wo_gate_iter_200.pth
wo_gate_iter_300.pth

wo_complement_iter_100.pth
wo_complement_iter_200.pth
wo_complement_iter_300.pth

full_iter_100.pth
full_iter_200.pth
full_iter_300.pth
```

---

## 10. 实验执行顺序

推荐按照以下顺序执行：

```text
Step 1：确认 Stage2-300 checkpoint 路径
Step 2：确认 Ours Full Stage3-300 结果是否已有
Step 3：运行 w/o Sinkhorn
Step 4：运行 w/o Gate
Step 5：运行 w/o Complement
Step 6：分别评估 detection 和 tracking 指标
Step 7：整理 mAP、NDS、AMOTA、Recall、MOTA、FP、IDS、FAF
Step 8：形成消融实验表格
Step 9：根据结果撰写模块贡献分析
```

---

## 11. 评估指标

每个实验均记录以下指标：

```text
mAP
NDS
AMOTA
Recall
MOTA
FP
IDS
FAF
```

其中重点分析：

```text
AMOTA：整体跟踪性能
MOTA：跟踪准确率
Recall：目标召回能力
FP：误检数量
IDS：身份切换次数
FAF：平均每帧误检率
```

建议保留 detection 指标 mAP 和 NDS，但论文分析重点放在 tracking 指标上。

---

## 12. 推荐消融实验表格

| Setting        | Sinkhorn | Gate | Complement | bbox fine-tune | mAP↑ | NDS↑ | AMOTA↑ | Recall↑ | MOTA↑ | FP↓ | IDS↓ | FAF↓ |
| -------------- | -------- | ---- | ---------- | -------------- | ---: | ---: | -----: | ------: | ----: | --: | ---: | ---: |
| w/o Sinkhorn   | ✗        | ✓    | ✓          | ✓              |      |      |        |         |       |     |      |      |
| w/o Gate       | ✓        | ✗    | ✓          | ✓              |      |      |        |         |       |     |      |      |
| w/o Complement | ✓        | ✓    | ✗          | ✓              |      |      |        |         |       |     |      |      |
| Ours Full      | ✓        | ✓    | ✓          | ✓              |      |      |        |         |       |     |      |      |

表格中建议加粗每列最优结果。

---

## 13. 结果分析重点

### 13.1 w/o Sinkhorn 分析重点

若 w/o Sinkhorn 的 AMOTA、MOTA 下降或 IDS 上升，说明 Sinkhorn soft matching 对跨车路 query 关联具有关键作用。

### 13.2 w/o Gate 分析重点

若 w/o Gate 的 FP、IDS 或 FAF 上升，说明 confidence-aware gate 能够抑制低置信度匹配造成的错误融合。

### 13.3 w/o Complement 分析重点

若 w/o Complement 的 Recall 下降或 FN 上升，说明 new-object complement 能够有效补充路侧可见但车端不足的目标。

### 13.4 Ours Full 分析重点

若完整方法在 AMOTA、MOTA、Recall 或 IDS/FAF 上取得较优结果，说明三个模块之间具有互补作用，能够共同提升车路协同跟踪性能。

---

## 14. 风险与应对

### 风险一：某个消融结果反而更好

如果某个消融版本在单一指标上超过完整方法，不要直接否定方法。需要看综合指标。例如：

```text
w/o Complement Recall 更低但 IDS 更好
w/o Gate Recall 更高但 FP 更高
w/o Sinkhorn FP 更低但 AMOTA 更差
```

论文中应解释为不同模块带来了不同指标之间的 trade-off。

### 风险二：完整方法不是所有指标最优

这是正常现象。本文重点是综合跟踪性能，不要求每个单项指标都最优。应重点强调 AMOTA、MOTA、Recall、IDS、FAF 的整体表现。

### 风险三：消融之间差距较小

如果差距较小，可以说明模型在 Stage3-300 时已进入局部稳定区，模块贡献体现在综合指标和趋势上，而不是单一指标的大幅变化。

---

## 15. 论文实验设置可用描述

为公平验证各模块的贡献，本文设计了三项消融实验，分别去除 Sinkhorn soft matching、confidence-aware gate 和 new-object complement。所有消融实验均从相同的 Stage2-300 checkpoint 初始化，并采用一致的 Stage3 微调策略训练 300 iterations。除被移除的目标模块外，其余网络结构、优化器参数、学习率设置、训练数据和评估协议均保持一致。该设置能够有效避免不同初始化和训练时长对实验结果造成干扰，从而更加直接地分析各模块对车路协同多目标跟踪性能的影响。

---

## 16. 最终计划总结

本次消融实验共包含三个核心变体：

```text
1. w/o Sinkhorn
2. w/o Gate
3. w/o Complement
```

并以 Ours Full 作为完整方法对照。

所有实验统一采用：

```text
Stage2-300 checkpoint → Stage3-300 iterations
```

最终通过 mAP、NDS、AMOTA、Recall、MOTA、FP、IDS 和 FAF 指标进行综合比较。

本消融实验的目标不是单独调优每一个变体，而是在严格公平的条件下证明本文三个核心模块分别对跨智能体匹配、错误融合抑制和新目标补充具有实际贡献。
