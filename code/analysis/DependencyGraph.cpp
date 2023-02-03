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

} // namespace rust_compiler::analysis::attributor
