#pragma once

#include <llvm/Target/TargetMachine.h>

namespace rust_compiler {

class Target {
  llvm::TargetMachine  *tm;

public:
  Target(llvm::TargetMachine  *tm);

  unsigned getVectorWidth();
  unsigned getPointerSizeInBits();
};

} // namespace rust_compiler
