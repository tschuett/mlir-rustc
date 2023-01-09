#pragma once

#include "Analysis/Attributer/DependencyGraphNode.h"
#include "Analysis/Attributer/IRPosition.h"

namespace rust_compiler::analysis::attributer {

class Attributer;

class AbstractElement : public IRPosition, public DependencyGraphNode {
public:
  virtual ~AbstractElement() = default;

  AbstractElement(const IRPosition &pos) : IRPosition(pos) {}

  virtual void initialize(Attributer &solver) {}
};

} // namespace rust_compiler::analysis::attributer
