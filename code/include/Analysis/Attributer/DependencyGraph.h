#pragma once

#include "Analysis/Attributer/Common.h"
#include "Analysis/Attributer/DependencyGraphNode.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SetVector.h>
#include <map>
#include <set>
#include <vector>

namespace rust_compiler::analysis::attributor {

class DepdendencyGraph {

public:
  void addDependency(const DependencyGraphNode *FromAA,
                     const DependencyGraphNode *ToAA, DepClass DepClass);

  std::vector<DependencyGraphNode *>
  getDependencies(const DependencyGraphNode *FromAA);

  bool areDepsEmpty(const DependencyGraphNode *Node);

  DependencyGraphNode *pickDependency(const DependencyGraphNode *Node);

  void removeDependency(const DependencyGraphNode *owner,
                        const DependencyGraphNode *dep);

  void resetNewNodes();
  llvm::SetVector<DependencyGraphNode *, std::vector<DependencyGraphNode *>,
                  llvm::DenseSet<DependencyGraphNode *>>
  getNewNodes();

private:
  // FIXME highly inefficient
  std::map<const DependencyGraphNode *, std::set<const DependencyGraphNode *>>
      dependendencies;
  llvm::SetVector<DependencyGraphNode *,
                  std::vector<DependencyGraphNode *>,
                  llvm::DenseSet<DependencyGraphNode *>>
      newNodes;
};

} // namespace rust_compiler::analysis::attributor
