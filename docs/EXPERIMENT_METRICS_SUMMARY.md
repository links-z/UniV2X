# 已有实验指标汇总

本文档记录当前工作区已有 eval log 中可直接追溯的关键结果，用于后续论文对照、消融和负结果分析。数值来自对应日志中的 pts_bbox_NuScenes 指标输出。

| 实验/日志 | 方法说明 | mAP | AMOTA | Recall | FP | IDS | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| eval_univ2x_coop_e2e.log.txt | 原始/当前 baseline 参考 | 0.0518 | 0.1040 | 0.1620 | 643 | 118 | mAP 稳定，但 IDS 高，tracking 有提升空间 |
| Zeval_compA_ft.log | learnable/Sinkhorn direct fusion | 0.0353 | 0.1013 | 0.1611 | 512 | 28 | IDS 降低明显，但 mAP 掉太多 |
| Zeval_compA_ft_v2.log | learnable/Sinkhorn direct fusion v2 | 0.0370 | 0.1024 | 0.1684 | 560 | 30 | 比 v1 略好，但 mAP 仍不可接受 |
| Zeval_compA_v3_gated.log | gated direct fusion | 0.0365 | 0.0981 | 0.1530 | 501 | 34 | 更保守但 AMOTA 和 mAP 都弱 |
| Zeval_compA_zeroshot.log | zero-shot direct fusion | 0.0365 | 0.0991 | 0.1625 | 521 | 37 | 无训练版本仍有 mAP 损伤 |
| eval_sinkhorn_detadapt_pilot.log | Sinkhorn det-adapt pilot | 0.0426 | 0.1119 | 0.2200 | 769 | 61 | Recall/AMOTA 提升，但 mAP 下降且 FP 偏高 |
| eval_pilot_sweep1.log | pilot sweep 1 | 0.0399 | 0.1101 | 0.2112 | 656 | 42 | IDS/FP 较稳，但 mAP 太低 |
| eval_pilot_sweep2.log | pilot sweep 2 | 0.0425 | 0.1122 | 0.2204 | 759 | 60 | 接近 det-adapt，FP 偏高 |
| eval_pilot_sweep3.log | pilot sweep 3 | 0.0420 | 0.1059 | 0.2050 | 643 | 45 | FP 控制好，但 AMOTA 提升不足 |
| Zeval_assoc_only_ep3.log | assoc-only/direct variant | 0.0382 | 0.0985 | 0.1540 | 507 | 29 | IDS 低，但主指标弱 |

## 当前判断

- direct query fusion 的共同问题是 mAP 明显下降。
- Sinkhorn/association 确实能改善 IDS 或 Recall，但 FP 与 mAP 没有同时控制好。
- 后续主线应切到 detection-preserved cooperative tracking：先恢复 mAP，再用 association/candidate pool 提升 AMOTA 和 IDS。

## 第一阶段验收标准

使用 projects/configs_e2e_univ2x/univ2x_coop_e2e_detpreserve.py 进行验证。

目标：

    mAP >= 0.050
    AMOTA 不低于 baseline 太多
    确认 direct fusion 已经不再污染 detection path
