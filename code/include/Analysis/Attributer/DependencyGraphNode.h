#pragma once

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/AsmState.h>

namespace rust_compiler::analysis::attributor {

class DependencyGraphNode {
public:
  virtual ~DependencyGraphNode() = default;

  virtual void print(llvm::raw_ostream &os, mlir::AsmState &asmState) const {
    os << "DepGraphNode unimpl\n";
  }
};

} // namespace rust_compiler::analysis::attributor
