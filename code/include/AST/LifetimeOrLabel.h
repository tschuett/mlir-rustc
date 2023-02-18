#pragma once

#include "AST/AST.h"

#include <string>

namespace rust_compiler::ast {

class LifetimeOrLabel : public Node {
  std::string label;

public:
  LifetimeOrLabel(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast
