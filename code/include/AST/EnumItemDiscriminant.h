#pragma once

#include "AST/AST.h"
#include "AST/Expression.h"

#include <memory>

namespace rust_compiler::ast {

class EnumItemDiscriminant : public Node {
  std::shared_ptr<Expression> expr;

public:
  EnumItemDiscriminant(Location loc) : Node(loc) {}

  void setExpression(std::shared_ptr<Expression> e) { expr = e; }
};

}; // namespace rust_compiler::ast
