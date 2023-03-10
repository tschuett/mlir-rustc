#include "Analysis/Loops.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallPtrSet.h>
// #include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Dominance.h>
#include <stack>

using namespace mlir;
using namespace llvm;

namespace rust_compiler::analysis {

/// https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
class StronglyConnectedComponent {
  SmallPtrSet<Block *, 8> graph;
  uint32_t index = 0;
  std::stack<Block *> stack;
  llvm::DenseMap<Block *, uint32_t> indexMap;
  llvm::DenseMap<Block *, uint32_t> lowLinkMap;
  llvm::DenseSet<Block *> onStack;

public:
  StronglyConnectedComponent(const SmallPtrSetImpl<Block *> &allBlocks) {
    graph = {allBlocks.begin(), allBlocks.end()};
  }

  bool isScc() {
    bool success = false;
    for (auto b : graph) {
      if (indexMap.count(b) == 0)
        success |= strongConnect(b);
    }
    return success;
  }

private:
  bool strongConnect(Block *v) {
    indexMap.insert({v, index});
    lowLinkMap.insert({v, index});
    ++index;
    stack.push(v);
    onStack.insert(v);

    for (auto succ : v->getSuccessors()) {
      if (graph.count(succ) == 1) {
        if (indexMap.count(succ) == 0) {
          // Successor succ has not yet been visited; recurse on it
          strongConnect(succ);
          uint32_t min =
              std::min(lowLinkMap.lookup(v), lowLinkMap.lookup(succ));
          lowLinkMap.insert({v, min});
        } else if (onStack.count(succ) == 1) {
          // Successor w is in stack S and hence in the current SCC
          // If w is not on stack, then (v, w) is an edge pointing to an SCC
          // already found and must be ignored

          // Note: The next line may look odd
          // - but is correct. It says w.index not w.lowlink; that is deliberate
          // and from the original paper
          uint32_t min =
              std::min(lowLinkMap.lookup(v), lowLinkMap.lookup(succ));
          lowLinkMap.insert({v, min});
        }
      }
    }
    // If v is a root node, pop the stack and generate an SCC
    if (lowLinkMap.lookup(v) == indexMap.lookup(v)) {
      // start a new strongly connected component
      mlir::Block *w = nullptr;
      SmallPtrSet<Block *, 8> scc;
      do {
        w = stack.top();
        stack.pop();
        onStack.erase(w);
        scc.insert(w);
      } while (w != v);
      // show cc
      if (scc.size() == graph.size()) {
        // yeah
        return true;
      } else {
        return false;
      }
    }
  }
};

llvm::Expected<std::vector<Loop>> detectLoop(mlir::func::FuncOp *f) {
  mlir::DominanceInfo domInfo;
  SmallPtrSet<Block *, 8> allBlocks;

  for (auto &block : f->getBody())
    allBlocks.insert(&block);

  for (auto &block : f->getBody()) {
    SmallPtrSet<Block *, 8> dominatedBlocks;
    for (auto &innerBlock : f->getBody()) {
      if (domInfo.dominates(&block, &innerBlock))
        dominatedBlocks.insert(&innerBlock);
    }
    if (dominatedBlocks.size() > 1) {
      // reachability check
      SmallPtrSet<Block *, 8> allLoopBlocks(dominatedBlocks);
      allLoopBlocks.insert(&block);
      StronglyConnectedComponent scc(allBlocks);
      if (scc.isScc()) {
        // xxx;
      }
    }
  }
}

} // namespace rust_compiler::analysis
