#include "Target.h"

#include <llvm/MC/MCRegisterInfo.h>

namespace rust_compiler {

using namespace llvm;

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
