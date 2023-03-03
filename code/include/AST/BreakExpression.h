#pragma once

#include "AST/AST.h"
#include "AST/Expression.h"
#include "AST/LifetimeOrLabel.h"

#include <memory>
#include <optional>

namespace rust_compiler::ast {

class BreakExpression : public ExpressionWithoutBlock {
  std::shared_ptr<Expression> expr;
  std::optional<LifetimeOrLabel> label;

public:
  BreakExpression(Location loc)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::BreakExpression){};

  void setExpression(std::shared_ptr<Expression>);
  void setLifetime(LifetimeOrLabel l) { label = l; }
};

} // namespace rust_compiler::ast
