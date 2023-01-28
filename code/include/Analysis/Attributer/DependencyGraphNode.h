#pragma once

#include "Analysis/Attributer/Common.h"

#include <llvm/ADT/TinyPtrVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/AsmState.h>

namespace rust_compiler::analysis::attributor {

class DependencyGraphNode {
public:
  using DepTy = llvm::PointerIntPair<DependencyGraphNode *, 1>;

  virtual ~DependencyGraphNode() = default;

  virtual void print(llvm::raw_ostream &os, mlir::AsmState &asmState) const {
    os << "DepGraphNode unimpl\n";
  }

  llvm::TinyPtrVector<DepTy> &getDeps() { return deps; }

  void addDependency(const DependencyGraphNode *FromAA, DepClass DepClass);

protected:
  llvm::TinyPtrVector<DepTy> deps;

  friend struct DependencyGrap;
};

} // namespace rust_compiler::analysis::attributor
