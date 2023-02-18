#pragma once

#include "AST/AST.h"

namespace rust_compiler::ast {

class LifetimeParam : public Node {
public:
  LifetimeParam(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast
