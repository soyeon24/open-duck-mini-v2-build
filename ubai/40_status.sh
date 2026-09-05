#!/bin/bash
# 학교 갈 때마다 상태 확인용. 게이트 노드에서 실행.
REPO="${REPO:-$HOME/Open_Duck_Playground}"
CKPT="$REPO/checkpoints"
echo "=== 큐 ==="
squeue -u "$USER" -o "%.10i %.12j %.9P %.8T %.10M %.10l %.6D %R"
echo
echo "=== 누적 진행 ==="
echo "progress: $(cat "$CKPT/.progress" 2>/dev/null || echo 0) steps"
[ -f "$CKPT/.done" ] && echo "*** 목표 달성 (.done) ***"
echo
echo "=== 체크포인트 / ONNX (최신 5개) ==="
ls -1t "$CKPT" 2>/dev/null | head -10
echo
echo "=== 최근 로그 꼬리 ==="
LAST=$(ls -1t "$REPO"/logs/*.out 2>/dev/null | head -1)
[ -n "$LAST" ] && { echo "-- $LAST"; grep -E "^STEP:|Saving checkpoint|Error|error" "$LAST" | tail -15; }
echo
echo "=== 최근 완료 잡 ==="
sacct -u "$USER" --format=JobID%12,JobName%12,State,Elapsed,ExitCode -X | tail -8
