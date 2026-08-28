
// --- torch 2.11 compat: restored c10::hip::*MasqueradingAsCUDA stream helpers ---
// The pip torch 2.11 nightly for rocm7.0 removed these; ROCm's TransformerEngine
// hipify still emits them. Verbatim from a known-good ROCm torch 2.11 source build.
#ifdef USE_ROCM
namespace c10 { namespace hip {
inline c10::cuda::CUDAStream getStreamFromPoolMasqueradingAsCUDA(
    const bool isHighPriority = false, DeviceIndex device = -1) {
  return c10::cuda::getStreamFromPool(isHighPriority, device);
}
inline c10::cuda::CUDAStream getStreamFromPoolMasqueradingAsCUDA(
    const int priority, DeviceIndex device = -1) {
  return c10::cuda::getStreamFromPool(priority, device);
}
inline auto& getStreamFromExternalMasqueradingAsCUDA =
    c10::cuda::getStreamFromExternal;
inline c10::cuda::CUDAStream getDefaultHIPStreamMasqueradingAsCUDA(
    DeviceIndex device_index = -1) {
  return c10::cuda::getDefaultCUDAStream(device_index);
}
inline c10::cuda::CUDAStream getCurrentHIPStreamMasqueradingAsCUDA(
    DeviceIndex device_index = -1) {
  return c10::cuda::getCurrentCUDAStream(device_index);
}
inline auto& setCurrentHIPStreamMasqueradingAsCUDA =
    c10::cuda::setCurrentCUDAStream;
}} // namespace c10::hip
#endif // USE_ROCM
