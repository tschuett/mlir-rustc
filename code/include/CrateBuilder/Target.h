#pragma once

#include <llvm/Target/TargetMachine.h>

namespace rust_compiler::crate_builder {

class Target {
  llvm::TargetMachine *tm;
  unsigned vectorWidth = 0;
  bool sve = false;
  llvm::StringMap<bool> features;

public:
  Target(llvm::TargetMachine *_tm);

  unsigned getVectorWidth();
  unsigned getPointerSizeInBits();
};

} // namespace rust_compiler::crate_builder
