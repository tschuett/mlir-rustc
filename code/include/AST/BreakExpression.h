#pragma once

#include "AST/AST.h"
#include "AST/Expression.h"

#include <memory>

namespace rust_compiler::ast {

class BreakExpression : public Node {
  std::shared_ptr<Expression> expr;

public:
  BreakExpression(Location loc) : Node(loc){};
};

} // namespace rust_compiler::ast


// FIXME LIFETIME_OR_LABEL
