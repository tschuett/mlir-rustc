#include "Analysis/Attributer/DependencyGraph.h"

namespace rust_compiler::analysis::attributor {

void DepdendencyGraph::addEdge(const DependencyGraphNode *FromAA,
                               const DependencyGraphNode *ToAA,
                               DepClass DepClass) {
  const_cast<DependencyGraphNode *>(ToAA)->addDependency(FromAA, DepClass);
}

} // namespace rust_compiler::analysis::attributor
