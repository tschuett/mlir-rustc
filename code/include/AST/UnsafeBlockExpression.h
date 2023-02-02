#pragma once

#include "AST/AST.h"
#include "AST/Expression.h"

#include <memory>

namespace rust_compiler::ast {

class UnsafeBlockExpression : public Node {
  std::shared_ptr<Expression> expr;

public:
  UnsafeBlockExpression(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast



// FIXME kind?
