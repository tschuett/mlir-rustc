#include "CrateBuilder/Target.h"

#include <llvm/TargetParser/Host.h>

namespace rust_compiler::crate_builder {

Target::Target(llvm::TargetMachine *_tm) {
  tm = _tm;

  llvm::StringMap<bool, llvm::MallocAllocator> _features;

  llvm::sys::getHostCPUFeatures(_features);

  features = _features;

  if (features.count("avx512") == 1)
    vectorWidth = 512;
  else if (features.count("avx") == 1)
    vectorWidth = 256;
  else if (features.count("sse2") == 1)
    vectorWidth = 128;
  else if (features.count("neon") == 1)
    vectorWidth = 128;
  else if (features.count("sve") == 1)
    sve = true;
}

} // namespace rust_compiler::crate_builder
