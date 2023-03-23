#include "Analysis/Cycles.h"

namespace rust_compiler::analysis {

void Cycles::depthFirstSearch(mlir::Block *block, uint32_t currentDepth) {
  // visit current node
  depths.insert({block, currentDepth});
  // left sutree
  // right tree
  for (auto *b : block->getSuccessors())
    depthFirstSearch(b, currentDepth + 1);
}

void Cycles::analyze(mlir::func::FuncOp *f) {
  fun = f;

  // first step: depth first search
  depthFirstSearch(&f->getRegion().front(), 0);

  for (unsigned i = 0; i < depths.size(); ++i) {
  }
}

} // namespace rust_compiler::analysis
