#pragma once

#include "AST/Expression.h"
#include "AST/LifetimeOrLabel.h"

#include <optional>

namespace rust_compiler::ast {

class ContinueExpression : public ExpressionWithoutBlock {
  std::optional<LifetimeOrLabel> label;

public:
  ContinueExpression(Location loc)
      : ExpressionWithoutBlock(
            loc, ExpressionWithoutBlockKind::ContinueExpression) {}

  void setLifetime(LifetimeOrLabel &orl) { label = orl;}
};

} // namespace rust_compiler::ast
