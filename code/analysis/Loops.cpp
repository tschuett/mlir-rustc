#include "Analysis/Loops.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Support/LLVM.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallPtrSet.h>
// #include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/IR/Block.h>
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

  void run() {
    for (auto b : graph)
      if (indexMap.count(b) == 0)
        strongConnect(b);
  }

  std::vector<llvm::SmallPtrSet<Block *, 8>> getSccs() { return sccs; }

private:
  std::vector<llvm::SmallPtrSet<Block *, 8>> sccs;

  void strongConnect(Block *v) {
    // Set the depth index for v to smallest unused index
    indexMap.insert({v, index});
    lowLinkMap.insert({v, index});
    ++index;
    stack.push(v);
    onStack.insert(v);

    // consider sucessors of v
    for (auto succ : v->getSuccessors()) {
      if (graph.count(succ) == 1) {
        if (indexMap.count(succ) == 0) {
          // Successor succ has not yet been visited; recurse on it
          strongConnect(succ);
          uint32_t min =
              std::min(lowLinkMap.lookup(v), lowLinkMap.lookup(succ));
          lowLinkMap.insert({v, min});
        } else if (onStack.count(succ) == 1) {
          // Successor w (succ) is in stack S and hence in the current SCC
          // If w (succ) is not on stack, then (v, w) is an edge pointing to an
          // SCC already found and must be ignored

          // Note: The next line may look odd
          // - but is correct. It says w.index not w.lowlink; that is deliberate
          // and from the original paper
          uint32_t min = std::min(lowLinkMap.lookup(v), indexMap.lookup(succ));
          lowLinkMap.insert({v, min});
        }
      }
    }

    // If v is a root node, pop the stack and generate an SCC
    if (lowLinkMap.lookup(v) == indexMap.lookup(v)) {
      // start a new strongly connected component
      mlir::Block *w = nullptr;
      SmallPtrSet<Block *, 8> nextScc;
      do {
        w = stack.top();
        stack.pop();
        onStack.erase(w);
        nextScc.insert(w);
      } while (w != v);
      // output the current stronly connected component
      sccs.push_back(nextScc);
    }
  }
};

bool LoopDetector::doSetsOverlap(llvm::SmallPtrSetImpl<Block *> &first,
                                 llvm::SmallPtrSetImpl<Block *> &second) {
  for (auto f : first)
    if (second.count(f))
      return true;

  for (auto s : second)
    if (first.count(s))
      return true;

  return false;
}

bool LoopDetector::doesSetContains(llvm::SmallPtrSetImpl<Block *> &first,
                                   llvm::SmallPtrSetImpl<Block *> &second) {
  for (auto s : second)
    if (first.count(s) == 0)
      return false;
  return true;
}

void LoopDetector::createLoop(llvm::SmallPtrSetImpl<Block *> &scc,
                              Block *header) {
  Loop l;
  l.setHeader(header);
  l.setBlocks(scc);

  llvm::SmallPtrSet<Block *, 8> backEdges;

  for (Block *b : scc)
    if (b != header)
      if (domInfo.dominates(b, header))
        backEdges.insert(b);

  // back edge?
  if (backEdges.size() > 0)
    l.setBackEdges(backEdges);
  else
    return;

  if (backEdges.size() == 1)
    l.setLatch(*(backEdges.begin()));

  if (header->getSinglePredecessor() != nullptr)
    l.setPreHeader(header->getSinglePredecessor());

  loops.push_back(l);
}

/// based on dominance and scc
void LoopDetector::detectLoopCandidates() {
  // for each block in the function
  for (auto &block : f->getBody()) {
    SmallPtrSet<Block *, 8> dominatedBlocks;
    for (auto &innerBlock : f->getBody())
      if (domInfo.dominates(&block, &innerBlock))
        dominatedBlocks.insert(&innerBlock);

    // check scc
    if (dominatedBlocks.size() > 1) {
      // reachability check
      dominatedBlocks.insert(&block);
      StronglyConnectedComponent scc(dominatedBlocks);
      scc.run();
      // xxx;
      std::vector<llvm::SmallPtrSet<Block *, 8>> sccs = scc.getSccs();
      for (unsigned i = 0; i < sccs.size(); ++i)
        createLoop(sccs[i], &block);
    }
  }
}

void LoopDetector::analyzeInductionVariable(Loop *l) {
  f->walk([&](mlir::memref::AllocOp allocaOp) {
    if (not l->containsBlock(allocaOp->getBlock())) {
      f->walk([&](mlir::memref::LoadOp loadOp) {
        if (l->containsBlock(loadOp->getBlock())) {
          f->walk([&](mlir::memref::StoreOp storeOp) {
            if (l->containsBlock(storeOp->getBlock())) {
              if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(&loadOp)) {
                if (auto store =
                        mlir::dyn_cast<mlir::memref::StoreOp>(&storeOp)) {
                  if (auto alloca =
                          mlir::dyn_cast<mlir::memref::AllocOp>(&allocaOp)) {
                    // now what
                    mlir::Value al = alloca->getResult();
                    load->getMemref() == al;
                    store->getMemref() == al;
                  }
                }
              }
            }
          });
        }
      });
    }
  });
}

void LoopDetector::analyzeRelationShips() {
  // xxx

  llvm::SmallPtrSet<Block *, 8> blocksWithCalls;
  f->walk([&](mlir::func::CallOp c) { blocksWithCalls.insert(c->getBlock()); });

  for (unsigned i = 0; i < loops.size(); ++i)
    for (unsigned j = i + 1; j < loops.size(); ++j) {
      if (loops[i].getHeader() == loops[j].getHeader()) {
        // now what
      }
    }

  for (unsigned i = 0; i < loops.size(); ++i) {
    for (unsigned j = i + 1; j < loops.size(); ++j) {
      llvm::SmallPtrSet<mlir::Block *, 8> first = loops[i].getBlocks();
      llvm::SmallPtrSet<mlir::Block *, 8> second = loops[j].getBlocks();
      if (doSetsOverlap(first, second)) {
        // now what
      }
      if (doesSetContains(first, second)) {
        // now what; different headers?
      }
      if (doesSetContains(second, first)) {
        // now what; different headers?
      }
    }
  }
}

std::optional<Function> LoopDetector::analyze(mlir::func::FuncOp *f) {
  this->f = f;

  detectLoopCandidates();

  analyzeRelationShips();
}

} // namespace rust_compiler::analysis
