#pragma once

#include "AST/Expression.h"
#include "AST/LifetimeOrLabel.h"

#include <optional>

namespace rust_compiler::ast {

class ContinueExpression : public ExpressionWithoutBlock {
  std::optional<LifetimeOrLabel> lifetimeOrLabel;

public:
  ContinueExpression(Location loc)
      : ExpressionWithoutBlock(
            loc, ExpressionWithoutBlockKind::ContinueExpression) {}
};

} // namespace rust_compiler::ast
