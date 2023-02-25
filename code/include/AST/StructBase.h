#pragma once

#include "AST/Expression.h"
#include "AST/PathExpression.h"

#include <memory>

namespace rust_compiler::ast {

class StructBase : public Node {
  std::shared_ptr<PathExpression> path;

public:
  StructBase(Location loc) : Node(loc) {}

  void setPath(std::shared_ptr<PathExpression> p) { path = p;}
};

} // namespace rust_compiler::ast
