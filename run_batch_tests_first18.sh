#!/bin/bash
# Batch runner for the first 18 tests (the fixed / expected-passing set).
# NOTE: uses the vllm/vime-rocm:nixl image by default — tests 9 (ppo_disaggregate)
# and 11 (external_pd) require the RIXL/UCX additions in that image. It is a
# superset of :latest, so it runs the other 16 fine too. Override with IMAGE=...
cd ~/vime
TESTS=(
  "test_qwen2.5_0.5B_async_short.py"
  "test_qwen2.5_0.5B_short.py"
  "test_qwen2.5_0.5B_fanout_short.py"
  "test_qwen2.5_0.5B_opd_vllm.py"
  "test_qwen2.5_0.5B_debug_rollout_then_train.py"
  "test_qwen2_5_0_5B_non_colocate_pp.py"
  "test_qwen3_4B_ppo.py"
  "test_qwen3_4B_ppo_train_critic_only.py"
  "test_qwen3_4B_ppo_disaggregate.py"
  "test_qwen3_4B_streaming_partial_rollout.py"
  "test_qwen3_4B_external_pd.py"
  "test_vllm_rollout.py"
  "test_external_vllm_engines.py"
  "test_empty_colocated_weight_bucket.py"
  "test_ppo_logprob_entropy_gpu.py"
  "test_qwen3_5_mtp_bridge_mapping.py"
  "test_qwen3_linear_attention_cu_seqlens.py"
  "test_release_train.py"
)
IMAGE="${IMAGE:-vllm/vime-rocm:nixl}"
LOG_DIR="$HOME/logs"
mkdir -p "$LOG_DIR"
RUN_TS=$(date +%Y%m%d_%H%M%S)
SUMMARY="$LOG_DIR/summary_first18_${RUN_TS}.txt"
# Per-test wall-clock cap. Exceeding it kills the container and skips to the
# next test (deterministic replacement for manual Ctrl-C on a hung run).
# test_qwen3_0.6B_parallel_check is NOT in this set, so 1200s is ample.
TIMEOUT_SECS="${TIMEOUT_SECS:-1200}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARNING: HF_TOKEN is not set. Public models still download but may rate-limit."
fi

echo "Batch(first18) $RUN_TS | image=$IMAGE | timeout=${TIMEOUT_SECS}s | ${#TESTS[@]} tests" | tee "$SUMMARY"

for t in "${TESTS[@]}"; do
  LOG_FILE="$LOG_DIR/${t%.py}_${RUN_TS}.log"
  CNAME="vime-${t%.py}"
  echo "=== $(date) Running: $t (log: $LOG_FILE) ==="
  # --name lets us kill the exact container on timeout. `timeout` signals the
  # docker CLI; we also explicitly `docker kill` so the daemon-side container
  # actually dies rather than orphaning.
  timeout --signal=TERM "${TIMEOUT_SECS}s" \
  docker run --rm \
    --name "$CNAME" \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --privileged \
    --security-opt seccomp=unconfined \
    --ipc=host --shm-size=16g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --ulimit nofile=1048576:1048576 \
    -e VIME_AMD_ROCM=1 -e VIME_TEST_DEVICE=rocm \
    -e VIME_SCRIPT_EXTERNAL_RAY=0 \
    -e VIME_TEST_ENABLE_INFINITE_RUN=false \
    -e VIME_TEST_USE_DEEPEP=0 \
    -e VIME_TEST_USE_FP8_ROLLOUT=0 \
    -e VIME_TEST_ENABLE_EVAL=1 \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $HOME/models:/root/models \
    -v $(pwd):/root/vime \
    -w /root/vime \
    --entrypoint bash \
    "$IMAGE" \
    /root/vime/run_test_in_container.sh "$t" \
    > "$LOG_FILE" 2>&1
  rc=$?
  # Make sure no container is left running for this test.
  docker kill "$CNAME" >/dev/null 2>&1
  if [ "$rc" -eq 0 ]; then
    status="PASSED"
  elif [ "$rc" -eq 124 ] || [ "$rc" -eq 143 ]; then
    status="TIMEOUT(${TIMEOUT_SECS}s)"
  else
    status="FAILED(rc=$rc)"
  fi
  echo "  → $status"
  printf '%-48s %s\n' "$t" "$status" | tee -a "$SUMMARY"
done

echo "=== $(date) Batch complete. Summary: $SUMMARY ==="
cat "$SUMMARY"
