#!/bin/bash
# Fine-grained gamma sweep around 1.5, fixed thr=0.3

CKPT="projects/work_dirs_e2e_univ2x/univ2x_coop_e2e_lowlr_finetune/epoch_3.pth"
CFG="projects/configs_e2e_univ2x/univ2x_coop_e2e_lowlr_finetune.py"
THR=0.3

GAMMAS=( 1.2  1.4  1.6 )

for gamma in "${GAMMAS[@]}"; do
  log="eval_gamma_fine_g${gamma}.log"
  if [ -f "${log}" ] && grep -q "NuScenes" "${log}"; then
    echo "=== SKIP gamma=${gamma} ==="
    continue
  fi
  echo "=== gamma=${gamma} → ${log} ==="
  FUSION_GAMMA=${gamma} FUSION_COMPLEMENT_THR=${THR} \
    PYTHONPATH=$(pwd) python tools/test.py \
      "${CFG}" "${CKPT}" \
      --eval bbox \
      2>&1 | tee "${log}"
done

echo ""
echo "===== SUMMARY ====="
for gamma in "${GAMMAS[@]}"; do
  log="eval_gamma_fine_g${gamma}.log"
  amota=$(grep -oP "'pts_bbox_NuScenes/amota': \K[0-9.]+" "${log}" 2>/dev/null | head -1)
  map=$(grep -oP "'pts_bbox_NuScenes/mAP': \K[0-9.]+" "${log}" 2>/dev/null | head -1)
  ids=$(grep -oP "'pts_bbox_NuScenes/ids': \K[0-9.]+" "${log}" 2>/dev/null | head -1)
  recall=$(grep -oP "'pts_bbox_NuScenes/recall': \K[0-9.]+" "${log}" 2>/dev/null | head -1)
  printf "gamma=%-4s  mAP=%-8s  AMOTA=%-10s  Recall=%-10s  IDS=%s\n" \
    "${gamma}" "${map}" "${amota}" "${recall}" "${ids}"
done
