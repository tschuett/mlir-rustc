#pragma once

#include "Analysis/Attributer/Common.h"

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/AsmState.h>

#include <variant>

namespace rust_compiler::analysis::attributor {

class DependencyGraphNode {
public:

  virtual ~DependencyGraphNode() = default;

  virtual void print(llvm::raw_ostream &os, mlir::AsmState &asmState) const {
    os << "DepGraphNode unimpl\n";
  }
protected:
  friend struct DependencyGraph;
};

} // namespace rust_compiler::analysis::attributor
