#pragma once

#include "Analysis/Attributer/Common.h"
#include "Analysis/Attributer/DependencyGraphNode.h"

namespace rust_compiler::analysis::attributor {

class DepdendencyGraph {

  class Edge {};

public:
  void addEdge(const DependencyGraphNode *FromAA,
               const DependencyGraphNode *ToAA, DepClass DepClass);

};

} // namespace rust_compiler::analysis::attributor
