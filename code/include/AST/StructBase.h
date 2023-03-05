#pragma once

#include "AST/Expression.h"

#include <memory>

namespace rust_compiler::ast {

class StructBase : public Node {
  std::shared_ptr<Expression> path;

public:
  StructBase(Location loc) : Node(loc) {}

  void setPath(std::shared_ptr<Expression> p) { path = p;}
};

} // namespace rust_compiler::ast
