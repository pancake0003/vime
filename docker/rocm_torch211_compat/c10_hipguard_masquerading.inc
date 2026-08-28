
// --- torch 2.11 compat: c10::hip::*GuardMasqueradingAsCUDA structs ---
// ROCm TransformerEngine hipify maps at::cuda::CUDAGuard -> c10::hip::HIPGuardMasqueradingAsCUDA
// etc. The pip torch 2.11 nightly dropped these; restore them (thin subclasses of the
// c10::cuda guards that our nightly still provides).
// TE includes only <c10/hip/HIPGuard.h> so it needs the structs here; aiter also includes
// <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h> (which defines them too) -> the shared
// VIME_HIP_MASQ_GUARDS_DEFINED sentinel guarantees exactly one definition across both.
#ifdef USE_ROCM
#ifndef VIME_HIP_MASQ_GUARDS_DEFINED
#define VIME_HIP_MASQ_GUARDS_DEFINED
namespace c10 { namespace hip {
struct HIPGuardMasqueradingAsCUDA final : public c10::cuda::CUDAGuard {
  using c10::cuda::CUDAGuard::CUDAGuard;
};
struct OptionalHIPGuardMasqueradingAsCUDA final : public c10::cuda::OptionalCUDAGuard {
  using c10::cuda::OptionalCUDAGuard::OptionalCUDAGuard;
};
struct HIPStreamGuardMasqueradingAsCUDA final : public c10::cuda::CUDAStreamGuard {
  using c10::cuda::CUDAStreamGuard::CUDAStreamGuard;
};
struct OptionalHIPStreamGuardMasqueradingAsCUDA final : public c10::cuda::OptionalCUDAStreamGuard {
  using c10::cuda::OptionalCUDAStreamGuard::OptionalCUDAStreamGuard;
};
struct HIPMultiStreamGuardMasqueradingAsCUDA final : public c10::cuda::CUDAMultiStreamGuard {
  using c10::cuda::CUDAMultiStreamGuard::CUDAMultiStreamGuard;
};
}} // namespace c10::hip
#endif // VIME_HIP_MASQ_GUARDS_DEFINED
#endif // USE_ROCM
