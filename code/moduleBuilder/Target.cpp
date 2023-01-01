#include "Target.h"

#include "llvm/TargetParser/Host.h"

#include <llvm/MC/MCRegisterInfo.h>

namespace rust_compiler {

using namespace llvm;

Target::Target(llvm::TargetMachine *_tm) {
  tm = _tm;

  if (sys::getHostCPUFeatures(features)) {
    if (features.count("avx512") == 1) {
      vectorWidth = 512;
    } else if (features.count("avx") == 1) {
      vectorWidth = 256;
    } else if (features.count("sse") == 1) {
      vectorWidth = 128;
    } else if (features.count("neon") == 1) {
      vectorWidth = 128;
    } else if (features.count("sve") == 1) {
      sve = true;
    }
  }
}

unsigned Target::getVectorWidth() {
  const MCRegisterInfo *registerInfo = tm->getMCRegisterInfo();

  unsigned maxWidth = 0;
  for (const llvm::MCRegisterClass registerClass : registerInfo->regclasses()) {
    maxWidth = std::max(maxWidth, registerClass.getSizeInBits());
  }

  return maxWidth;
}

unsigned Target::getPointerSizeInBits() { return tm->getPointerSizeInBits(0); }

} // namespace rust_compiler
