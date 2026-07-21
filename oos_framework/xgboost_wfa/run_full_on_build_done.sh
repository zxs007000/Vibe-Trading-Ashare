#!/bin/bash
# 全量流水线启动器 (端到端无人值守)
#   1) 等待东财驱动写 BUILD_DONE (三表齐全: income+cash+balance)
#   2) 运行全量 v4 原型 (CPU 重活, 约 1h):
#        按落盘 booster_fold*.json 数量校验, 未达 4 折则自动重跑(封顶 4 次, 抗休眠)
#   3) 运行修正版 SHAP 分析 (读 4 折 OOS 子样本, 跨折方向一致性)
# 注: cnstock(data.cnstock.com) 已被我们的高并发爬取触发整域 403 封禁, 暂不可用;
#     基本面改回东财三大表派生(见 xgb_wfa_proto_v4_full.py load_fundamentals)。
# 用法: setsid bash run_full_on_build_done.sh
cd /workspace/quant_proto
LOG=/workspace/quant_proto/proto_v4_full_launcher.log
LAKE=${STOCKLAKE:-/workspace/stocklake}
BASE=/workspace/quant_proto
OUT=${WFA_OUT:-/workspace/quant_proto/v4proto_out}
N_FOLDS=4
MAX_V4_RETRY=4

booster_cnt() {
  ls "$OUT"/booster_fold*.json 2>/dev/null | wc -l
}

echo "$(date) [launcher] 等待 BUILD_DONE (东财三表齐全) ..." >> "$LOG"

while [ ! -f "$LAKE/BUILD_DONE" ]; do
  sleep 30
done
echo "$(date) [launcher] 检测到 BUILD_DONE ✅" >> "$LOG"

echo "$(date) [launcher] 开始全量 v4 (封顶重跑 $MAX_V4_RETRY 次)" >> "$LOG"

attempt=0
while [ "$attempt" -lt "$MAX_V4_RETRY" ]; do
  attempt=$((attempt+1))
  echo "$(date) [launcher] >>> v4 第 $attempt/$MAX_V4_RETRY 次运行" >> "$LOG"
  python3.11 xgb_wfa_proto_v4_full.py >> /workspace/quant_proto/proto_v4_full.log 2>&1
  rc=$?
  n=$(booster_cnt)
  echo "$(date) [launcher] v4 退出 rc=$rc booster数=$n/$N_FOLDS" >> "$LOG"
  if [ "$n" -ge "$N_FOLDS" ]; then
    echo "$(date) [launcher] v4 完成 ✅ (4 折 booster 已落盘)" >> "$LOG"
    break
  fi
  echo "$(date) [launcher] booster 不足 $N_FOLDS, 30s 后重跑" >> "$LOG"
  sleep 30
done

if [ "$(booster_cnt)" -lt "$N_FOLDS" ]; then
  echo "$(date) [launcher] ⚠️ v4 多次重跑仍未齐 4 折, 中止 SHAP, 需人工排查" >> "$LOG"
  exit 1
fi

echo "$(date) [launcher] >>> 运行修正版 SHAP 分析 (4 折跨折一致性)" >> "$LOG"
python3.11 shap_analysis.py --base "$OUT" >> /workspace/quant_proto/shap_full.log 2>&1
rc=$?
echo "$(date) [launcher] SHAP 退出 rc=$rc" >> "$LOG"
echo "$(date) [launcher] 🎉 全量流水线结束" >> "$LOG"
