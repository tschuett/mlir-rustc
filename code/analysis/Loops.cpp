#include "Analysis/Loops.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Dominance.h>
#include <mlir/Support/LLVM.h>
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

bool Loop::contains(mlir::Block *b) const { return loop.count(b) == 1; }

void Loop::findLoopPredecessor() {
  Block *out;
  for (auto pred : header->getPredecessors()) {
    if (!contains(pred)) { // the block is not in the loop
      if (out && out != pred)
        return;
      out = pred;
    }
  }

  loopPredecessor = out;
}

void Loop::findPreheader() {
  Block *out = loopPredecessor;
  if (!out)
    return;

  unsigned cnt = 0;
  for (auto succ : out->getSuccessors())
    ++cnt;
  if (cnt != 1)
    return;
  preHeader = out;
}

void Loop::findExitingBlocks() {
  for (auto BB : loop)
    for (auto *succ : BB->getSuccessors())
      if (!containsBlock(succ)) {
        exitingBlocks.insert(BB);
        break;
      }
}

void Loop::findExitBlocks() {
  for (auto BB : loop)
    for (auto *succ : BB->getSuccessors())
      if (!containsBlock(succ))
        exitBlocks.insert(succ);
}

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

  l.findExitBlocks();
  l.findExitingBlocks();
  l.findLoopPredecessor();
  l.findPreheader();

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

/// https://github.com/llvm/llvm-project/blob/82ac02e4a86070cf9924c245ff340aba1f62b45b/llvm/lib/Analysis/LoopInfo.cpp#L150
void LoopDetector::analyzeInductionVariable(Loop *l) {
  //  f->walk([&](mlir::memref::AllocOp allocaOp) {
  //    if (not l->containsBlock(allocaOp->getBlock())) {
  //      f->walk([&](mlir::memref::LoadOp loadOp) {
  //        if (l->containsBlock(loadOp->getBlock())) {
  //          f->walk([&](mlir::memref::StoreOp storeOp) {
  //            if (l->containsBlock(storeOp->getBlock())) {
  //              if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(&loadOp))
  //              {
  //                if (auto store =
  //                        mlir::dyn_cast<mlir::memref::StoreOp>(&storeOp)) {
  //                  if (auto alloca =
  //                          mlir::dyn_cast<mlir::memref::AllocOp>(&allocaOp))
  //                          {
  //                    // now what
  //                    mlir::Value al = alloca->getResult();
  //                    load->getMemref() == al;
  //                    store->getMemref() == al;
  //                  }
  //                }
  //              }
  //            }
  //          });
  //        }
  //      });
  //    }
  //  });
}

/// canonical 5 nested loops: how to detect? and precise nesting
void LoopDetector::analyzeLoopNests() {
  std::vector<std::pair<unsigned, unsigned>> nestingCandidates;
  // llvm::SmallSet<unsigned, 8> topLevelCandidates;

  llvm::SmallMapVector<unsigned, uint32_t, 8> levels;

  // analyze nested loops (expansive)
  for (unsigned i = 0; i < loops.size(); ++i) {
    for (unsigned j = i + 1; j < loops.size(); ++j) {
      llvm::SmallPtrSet<mlir::Block *, 8> first = loops[i].getBlocks();
      llvm::SmallPtrSet<mlir::Block *, 8> second = loops[j].getBlocks();
      if (loops[i].getHeader() != loops[j].getHeader()) {
        if (first.size() > second.size()) {
          if (doesSetContains(first, second)) {
            nestingCandidates.push_back({i, j});
            // topLevelCandidates.insert(i);
            levels.insert({i, 1});
            levels.insert({j, 0});
          }
        } else if (second.size() > first.size()) {
          if (doesSetContains(second, first)) {
            nestingCandidates.push_back({j, i});
            // topLevelCandidates.insert(j);
            levels.insert({j, 1});
            levels.insert({i, 0});
          }
        }
      }
    }
  }

  /// ??? LoopNest LoopLevel?
  bool changed = false;
  do {
    for (auto &par : nestingCandidates) {
      auto [l, r] = par;
      uint32_t rightLevel = levels.lookup(r);
      uint32_t leftLevel = levels.lookup(l);
      if (rightLevel > leftLevel) {
        levels[l] = rightLevel + 1;
        changed = true;
      }
    }
  } while (changed);
  // terminates?

  for (auto kv : levels) {
    loops[kv.first].level = kv.second;
  }

  // If loop has no parents, then it is head of loop nest

  // Everybody is head of loop nest until proven otherwise

  llvm::SmallVector<unsigned> head;
  for (auto par : nestingCandidates) {
    auto [l, r] = par;
    uint32_t rightLevel = levels.lookup(r);
    uint32_t leftLevel = levels.lookup(l);
  }
}

// Todo: getLevel() and getParent()

void LoopDetector::analyzeRelationShips() {
  // xxx

  //
  //      if (not doSetsOverlap(first, second) &&
  //          loops[i].getHeader() != loops[j].getHeader()) {
  //        // now what; -> they are disjoint
  //      }
  //      //      if (doSetsOverlap(first, second)) {
  //      //        // now what
  //      //      }
  //      if (doesSetContains(first, second) &&
  //          loops[i].getHeader() != loops[j].getHeader()) {
  //        // now what; second inner loop of first
  //      }
  //      if (doesSetContains(second, first) &&
  //          loops[i].getHeader() != loops[j].getHeader()) {
  //        // now what; first inner loop of second
  //      }

  llvm::SmallPtrSet<Block *, 8> blocksWithCalls;
  f->walk([&](mlir::func::CallOp c) { blocksWithCalls.insert(c->getBlock()); });

  for (unsigned i = 0; i < loops.size(); ++i)
    for (unsigned j = i + 1; j < loops.size(); ++j) {
      if (loops[i].getHeader() == loops[j].getHeader()) {
        // now what
      }
    }
}

std::optional<Function> LoopDetector::analyze(mlir::func::FuncOp *f) {
  this->f = f;

  detectLoopCandidates();

  analyzeRelationShips();
}

} // namespace rust_compiler::analysis

/*

  There are several *LoopNest* s

 */

void collectLops(mlir::func::FuncOp *f) {
  mlir::DominanceInfo dom;
  auto &DomTree = dom.getDomTree(&f->getRegion());
  // for (auto DomNode : post_order(DomTree)) {
  // }
}
