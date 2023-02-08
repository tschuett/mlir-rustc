#include "Analysis/Attributer/DependencyGraph.h"

namespace rust_compiler::analysis::attributor {

void DepdendencyGraph::addDependency(const DependencyGraphNode *FromAA,
                                     const DependencyGraphNode *ToAA,
                                     DepClass DepClass) {
  auto it = dependendencies.find(FromAA);
  if (it != dependendencies.end()) {
    it->second.insert(ToAA);
  } else {
    std::set<const DependencyGraphNode *> empty;
    empty.insert(ToAA);
    auto pair = dependendencies.insert({FromAA, empty});
  }
}

void DepdendencyGraph::resetNewNodes() { newNodes.clear(); }

llvm::SetVector<DependencyGraphNode *, std::vector<DependencyGraphNode *>,
                llvm::DenseSet<DependencyGraphNode *>>
DepdendencyGraph::getNewNodes() {
  return newNodes;
}

bool DepdendencyGraph::areDepsEmpty(const DependencyGraphNode *Node) {
  return dependendencies[Node].empty();
}

DependencyGraphNode *
DepdendencyGraph::pickDependency(const DependencyGraphNode *Node) {
  assert(false);
}

void DepdendencyGraph::removeDependency(const DependencyGraphNode *owner,
                                        const DependencyGraphNode *dep) {
  if (dependendencies.contains(owner)) {
    auto it = dependendencies.find(owner);
    it->second.erase(dep);
  }
}

} // namespace rust_compiler::analysis::attributor
