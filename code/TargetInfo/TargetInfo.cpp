#include "TargetInfo/TargetInfo.h"

namespace rust_compiler::target_info {

std::unique_ptr<TargetInfo>
getTargetInfo(llvm::Triple triple, std::span<std::string> cpuFeatureFlags) {

  return nullptr;
  
}

} // namespace rust_compiler::target_info
