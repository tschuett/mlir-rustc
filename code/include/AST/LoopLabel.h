#pragma once

#include "AST/AST.h"
#include "AST/LifetimeOrLabel.h"

namespace rust_compiler::ast {

class LoopLabel : public Node {
  LifetimeOrLabel label;

public:
  LoopLabel(Location loc) : Node(loc), label(loc) {}

  std::string getName() const { return label.getLabel(); }
};

} // namespace rust_compiler::ast
