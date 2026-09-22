# ROCm CI Status & Failure Investigation

Maintained tracker for running the vime GPU CI suites on AMD ROCm. Update this
file as tests are (re)run and root causes are confirmed or fixed.

## Ground rules for fixes

- **Only touch ROCm-specific files**: `.buildkite/pipeline-rocm.yaml`,
  `docker/Dockerfile.rocm`, and ROCm test variants (`tests/*-rocm.py`) or the
  runner scripts (`run_test_in_container.sh`, `run_*_main.sh`).
- **Do NOT modify** shared vime Python: no `vime/**` (utilities, `actors.py`,
  `arguments.py`, accelerator, ray, backends, …), and no shared CI utilities
  such as `tests/ci/gpu_lock_exec.py`.
- **Keep `torch.compile` ON.** Do not set `enforce_eager` / disable
  `VLLM_COMPILE` / turn off inductor as a workaround.

## Environment

- Host: single node, 8× AMD Instinct **MI350X** (gfx950), ~252 GiB VRAM each.
- Image: `rocm/pytorch-private:vime-09-08` (torch 2.12 / HIP 7.2, vLLM
  `0.28.1rc1.dev516`, vime + Megatron-LM baked in).
- Branch under test: `main` (upstream-synced; CUDA-named test files run on the
  ROCm runtime path — there are no `-rocm.py` variants on `main`).
- How tests are launched: one container per test via `run_test_in_container.sh`,
  which runs `tests/ci/gpu_lock_exec.py --count N --target-env-name
  HIP_VISIBLE_DEVICES -- python tests/<file>`. `--network host` is required
  (host docker bridge `docker0` is broken).
- Logs live under `rocm_test_logs/{,vllm-config/,megatron/}`.

## Results

### short suite (4 GPU each)

| Test | Exit | Family | Verdict |
|---|---|---|---|
| test_qwen3.5_0.8B_gsm8k_async_short.py | 1 | D | soft-fail; vLLM rollout engine crash |
| test_qwen3.5_0.8B_gsm8k_short.py | 1 | D | soft-fail; HIP device-side assert |
| test_qwen2.5_0.5B_fully_async_short.py | 0 | — | **PASS ✅** (gating test) |

### vllm-config suite (8 GPU each)

| Test | Exit | Family | Verdict |
|---|---|---|---|
| test_qwen2.5_0.5B_vllm_config.py | 1 | C | vLLM engine OOM at init |
| test_qwen2.5_0.5B_vllm_config_distributed.py | 1 | A | invalid device ordinal |
| test_vllm_config_mixed_offload.py | 1 | A | invalid device ordinal |
| test_vllm_config_mixed_offload_ft.py | 1 | A | invalid device ordinal |

### megatron suite (11 of 21 entries ran; #12 interrupted, #13–21 not run)

| Test | GPU / env | Exit | Family | Verdict |
|---|---|---|---|---|
| test_full_disk_weight_update.py | 4 | 0 | — | **PASS ✅** |
| test_quick_start_glm4_9B.py | 8 (4+4) | 1 | A | invalid device ordinal |
| test_glm4.7_30B_A3B_pd_mooncake.py | 8 | 1 | B | vLLM worker segfault at init |
| test_qwen3_30B_A3B.py | 8 (DEEPEP+FP8) | 1 | B | vLLM worker segfault at init |
| test_qwen3.6_35B_A3B_pd_mooncake.py | 8 (DEEPEP) | 1 | B | vLLM worker segfault at init |
| test_qwen3_30B_A3B_r3.py (cfg1) | 8 (DEEPEP+FP8, EVAL=0) | 1 | B | vLLM worker segfault at init |
| test_qwen3_30B_A3B_r3.py (cfg2) | 8 (EVAL=0) | 1 | B′ | vLLM worker: unsupported device GEMM |
| test_qwen3_4B_ppo.py | 8 | 1 | A | invalid device ordinal |
| test_qwen3_4B_ppo_disaggregate.py | 8 (4+4) | 1 | A | invalid device ordinal |
| test_qwen3_4B_ppo_train_critic_only.py | 8 | 1 | A | invalid device ordinal |
| test_ppo_logprob_entropy_gpu.py | 2 | 1 | E | numerical assertion (tensors not close) |
| test_release_train.py | 4 | — | — | interrupted mid-run (session teardown); rerun |
| test_qwen3_4B_streaming_partial_rollout.py | 8 | — | — | not run |
| test_moonlight_16B_A3B.py | 8 | — | — | not run |
| test_moonlight_16B_A3B_r3.py | 8 (EVAL=0) | — | — | not run |
| test_mimo_7B_mtp_only_grad.py | 8 | — | — | not run |
| test_qwen2.5_0.5B_debug_rollout_then_train.py | 8 | — | — | not run |
| test_qwen2.5_0.5B_opd_vllm.py | 8 | — | — | not run |
| test_qwen2.5_0.5B_fanout_short.py | 4 | — | — | not run |
| test_qwen2.5_0.5B_debug_train_dump_e2e.py | 8 | — | — | not run |
| test_qwen3_4B_external_pd.py | 6 (UPDATE_MODE=delta) | — | — | not run |

### Standalone sanity run (not a CI suite)

| Workflow | GPU | Result |
|---|---|---|
| Qwen3-8B colocate GRPO (`scripts/run-qwen3-8B-amd.sh`, `NUM_ROLLOUT=100`, GPUs 6,7) | 2 | **PASS ✅** exit 0 — full rollout→ref-logprob→train→weight-update loop, reward trended up, `train_rollout_logprob_abs_diff ≈ 0.014` (healthy) |

> Note: this run needed the supported `--disable-wandb-random-suffix` flag —
> the image ships `wandb 0.29.0` which moved `generate_id` out of `wandb.util`,
> and `vime/observability/wandb_utils.py:46` calls the old API. Real fix belongs
> in vime (out of scope here); the flag is the ROCm-side workaround.

## Failure families — root cause & candidate ROCm-only remediation

### Family A — `invalid device ordinal` (HIP/CUDA visible-device collision)

**Affected:** vllm_config_distributed, mixed_offload, mixed_offload_ft,
quick_start_glm4_9B, qwen3_4B_ppo, qwen3_4B_ppo_disaggregate,
qwen3_4B_ppo_train_critic_only.

**Signature:**
```
File "vime/ray/train_actor.py", line 56, in init
  accelerator.set_device(local_rank)
torch.AcceleratorError: CUDA error: invalid device ordinal
GPU device may be out of range, do you have enough GPUs?
```

**Root cause:** `train_actor.py` sets `LOCAL_RANK = get_local_gpu_id()` then
`accelerator.set_device(local_rank)`. It relies on Ray remapping each actor's
GPUs via `CUDA_VISIBLE_DEVICES` (there is even a TODO in the file noting the
`CUDA_VISIBLE_DEVICES` conflict). On ROCm, `gpu_lock_exec` exports
`HIP_VISIBLE_DEVICES=0..7` globally; `HIP_VISIBLE_DEVICES` takes precedence over
Ray's per-actor `CUDA_VISIBLE_DEVICES`, so the remap Ray depends on is defeated
and `set_device(local_rank)` indexes a device outside the actor's real visible
set. Confirmed by vLLM's repeated warning `Using CUDA_VISIBLE_DEVICES on ROCm is
deprecated … Please use HIP_VISIBLE_DEVICES instead`. Single-group 4-GPU colocate
tests (fully_async, full_disk_weight_update) do not trip it.

**Mechanism (refined by Experiment 1 + 1b):** Ray does per-actor GPU isolation on
ROCm by rewriting a `*_VISIBLE_DEVICES` env var per worker. The CI harness
(`gpu_lock_exec`) pins one such var globally to all acquired GPUs. The two then
collide:
- Pin `HIP_VISIBLE_DEVICES=0..7` (current `pipeline-rocm.yaml`): Ray remaps the
  actor's visible set to a subset, but vime's `LOCAL_RANK`/`get_local_gpu_id()`
  still yields a global index → `set_device()` indexes outside the actor's
  visible set → `invalid device ordinal`.
- Pin `CUDA_VISIBLE_DEVICES=0..7` (Experiment 1): gets *past* `set_device`, but
  Ray sets `HIP_VISIBLE_DEVICES='0'` on the rollout worker while the global
  `CUDA_VISIBLE_DEVICES='0..7'` lingers, and **vLLM's own**
  `rocm.py:_sync_hip_cuda_env_vars()` raises
  `ValueError: Inconsistent GPU visibility env vars … Please set only one, or
  ensure they match.` So the naive swap just trades one error for another.

**Candidate ROCm-only fix (Experiment 1b — the long-standing fix):** stop Ray from
rewriting the device var per worker so every actor inherits the single global pin
(all 8 GPUs visible), keeping `set_device(local_rank)` in range and leaving only
one visibility var set. Set, in `pipeline-rocm.yaml` / the runner env (no vime
code):
`RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1`,
`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`,
`RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1`, with `gpu_lock_exec` keeping the
`HIP_VISIBLE_DEVICES` target. **Do not** patch `train_actor.py`.

**Evidence log:**
- Exp 1 (`--target-env-name CUDA_VISIBLE_DEVICES`): cleared `invalid device
  ordinal` (0 occurrences), reached rollout-router launch + actor creation, then
  failed on vLLM's `_sync_hip_cuda_env_vars` consistency guard.
- Exp 1b (`RAY_EXPERIMENTAL_NOSET_{HIP,CUDA,ROCR}_VISIBLE_DEVICES=1`, HIP pin
  kept): **cleared Family A** — 0 `invalid device ordinal`, 0 visibility-guard
  errors. Ray no longer rewrites the per-worker device var, so every actor sees
  the global `HIP_VISIBLE_DEVICES=0..7` and `set_device(local_rank)` stays in
  range. The run then reached vLLM weight-transfer setup and hit a **new,
  downstream** failure (see Family G), not a Family-A regression.

Status: **Family A FIXED by the Ray NOSET env vars** (ROCm-only, no vime code).
**Applied** in `.buildkite/pipeline-rocm.yaml` (three `-e RAY_EXPERIMENTAL_NOSET_*`
on the shared `&rocm_gpu_test` docker-run anchor) — the pushed CI path. The local
helper `run_test_in_container.sh` also exports them for local runs, but that file
is not committed, so the pipeline is the source of truth. (A `Dockerfile.rocm`
`ENV` was considered but rejected: it would apply image-wide and change the
known-good AMD-script 8B path; the pipeline scoping is safer.) Remaining blocker
for the split actor+rollout placements is Family G (below).

### Family B — vLLM worker **segfault** at EngineCore init (big MoE models)

**Affected:** glm4.7_30B_A3B_pd_mooncake, qwen3_30B_A3B, qwen3.6_35B_A3B_pd_mooncake,
qwen3_30B_A3B_r3 (cfg1). All 4 logs show 6–8 segfaults.

**Signature:** during `EngineCore` worker init the process hard-segfaults inside
`_local_scalar_dense_cuda` (a tensor→scalar `.item()` / `bool()` device sync):
```
!!!!!!! Segfault encountered !!!!!!!
  at::native::_local_scalar_dense_cuda(at::Tensor const&)
  torch::autograd::THPVariable_bool_scalar → PyObject_IsTrue
…
EngineCore failed to start.
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```
Preceding worker log: `Found incompatible backend(s) [TURBOQUANT] with
AttentionType.DECODER. Overriding with ROCM_ATTN` (FP8 variants). Not a clean
Python exception — a native crash in the vLLM/torch MoE(+FP8) init path on ROCm.

**Candidate ROCm-only fix (needs investigation):** image/config level only —
e.g. `docker/Dockerfile.rocm` (kernel/lib versions: aiter, CK/hipBLASLt, deep_ep)
and/or ROCm test variants that select a working MoE/quant path. `deep_ep is not
installed` in-container, and these configs request
`--vllm-all2all-backend deepep_high_throughput` / `--vllm-enable-expert-parallel`.
**Constraint:** keep `torch.compile` on — do not "fix" via `enforce_eager`.
Status: **root cause identified (native segfault), remediation open.**

### Family B′ — vLLM worker: unsupported device GEMM (r3 cfg2, non-FP8)

**Affected:** qwen3_30B_A3B_r3 (cfg2, `ENABLE_EVAL=0`, no DEEPEP/FP8).

**Signature:**
```
ERROR multiproc_executor.py:1055 RuntimeError: wrong! device_gemm with the
specified compilation parameters does not support this GEMM problem
```
Same model family as B but the non-FP8/non-DeepEP path hits an unsupported
CK/aiter GEMM shape instead of segfaulting. Likely the same underlying
MoE-on-ROCm gap. Remediation space: `Dockerfile.rocm` kernel libs / ROCm test
variant. Status: **open.**

### Family C — vLLM engine **OOM** at init (colocate memory not freed)

**Affected:** vllm_config.py.

**Signature:**
```
ValueError: Free memory on device cuda:0 (60.65/251.98 GiB) on startup is less
than desired GPU memory utilization (0.7, 176.39 GiB).
```
Colocated Megatron actors still held ~191 GiB on GPU 0 when vLLM tried to reserve
0.7 → engine init aborts. Training memory wasn't offloaded/freed before rollout
on the ROCm path.

**Candidate ROCm-only fix (needs investigation):** a ROCm test variant could
lower `--vllm-gpu-memory-utilization` or adjust the offload knobs for this case.
Do not change core offload logic in `vime/**`. Status: **open.**

### Family D — gsm8k short rollout instability (known soft-fail)

**Affected:** gsm8k_async_short, gsm8k_short. Already `soft_fail: true` in
`pipeline-rocm.yaml`.
- async: vLLM `EngineCore` raised an exception Ray couldn't deserialize
  (`UnserializableException`) → `/generate` returned HTTP 500 for all 60 retries.
- sync: HIP **device-side assertion** during `train.py`.

Status: **known non-blocking**; distinct symptoms, both on the vLLM/ROCm rollout
path. Pipeline comment cites NaN-logprob divergence, but observed symptoms differ.

### Family E — numerical assertion (ppo_logprob_entropy_gpu, 2 GPU)

**Affected:** ppo_logprob_entropy_gpu.

**Signature:** `AssertionError: Tensor-likes are not close!`
(`tests/test_ppo_logprob_entropy_gpu.py:173`). A logprob/entropy reference
comparison exceeds tolerance on ROCm — no device-mapping or engine issue.

**Candidate ROCm-only fix:** this is a test-file assertion; a ROCm test variant
could widen the tolerance for the ROCm numerics (allowed — test file). Confirm
the gap is precision, not a real correctness bug, first. Status: **open.**

### Family G — RCCL weight-transfer process-group init fails (surfaced after Family A fix)

**Affected:** vllm_config_distributed (seen once the Ray NOSET fix clears Family A;
likely affects the other split actor+rollout placements too).

**Signature:**
```
vllm/distributed/weight_transfer/nccl_common.py:158 in stateless_init_process_group
vllm/distributed/weight_transfer/nccl_common.py:213 in worker_init_process_group
RuntimeError: NCCL error: unhandled cuda error (run with NCCL_DEBUG=INFO for details)
Exception: Call to collective_rpc method failed: NCCL error: unhandled cuda error
… shm_broadcast: No available shared memory broadcast block found in 60 seconds
```

**Root cause (open):** the stateless RCCL communicator that syncs trained weights
from the Megatron actor into the vLLM rollout engine fails to initialize on this
ROCm image for the 4-actor + 4-rollout split. Colocate 2-GPU weight transfer
(nccl backend) works (Qwen3-8B run), so this is specific to the cross-group
stateless PG on ROCm. Needs `NCCL_DEBUG=INFO` to get the real RCCL error.

**Candidate ROCm-only remediation (to try):** RCCL env tuning in
`pipeline-rocm.yaml` / runner env — e.g. `NCCL_DEBUG=INFO` (diagnose), then
`NCCL_P2P_DISABLE=1` / `NCCL_SHM_DISABLE=1` / interface pinning as needed. Keep
`torch.compile` on. Status: **open; next after Family A rollout.**

## Known-good on ROCm

- 4-GPU single-group colocate: `fully_async_short`, `full_disk_weight_update`.
- 2-GPU colocate GRPO via the AMD script (bypasses `gpu_lock_exec` /
  HIP_VISIBLE_DEVICES; sets `VISIBLE_GPUS` itself) — end-to-end healthy.
- Common thread: failures cluster on ≥8-GPU multi-actor placements (Family A) and
  large MoE/quant rollout init (Families B/B′). The AMD-script path avoiding the
  `gpu_lock_exec` HIP_VISIBLE_DEVICES pin is the strongest hint for Family A.

## Open experiments / next steps

1. **Family A validation** — DONE (Exp 1): the `CUDA_VISIBLE_DEVICES` swap
   cleared `invalid device ordinal` but exposed vLLM's HIP/CUDA consistency
   guard, proving the two-visibility-var collision. **Exp 1b (in progress):**
   keep the `HIP_VISIBLE_DEVICES` pin and add
   `RAY_EXPERIMENTAL_NOSET_{HIP,CUDA,ROCR}_VISIBLE_DEVICES=1` so Ray stops
   per-worker device rewriting; validate on `vllm_config_distributed`, then roll
   into `pipeline-rocm.yaml` if green.
2. **Family B triage** — capture full native backtrace / try image-level
   kernel-lib options in `Dockerfile.rocm`; check whether installing `deep_ep`
   or changing the ROCm MoE path clears the segfault (torch.compile stays on).
3. **Family C** — retry `vllm_config` with a lower rollout mem-util in a ROCm
   test variant.
4. **Family E** — inspect the tolerance at `test_ppo_logprob_entropy_gpu.py:173`.
5. Finish the megatron suite (rerun `release_train`, then #13–21).
