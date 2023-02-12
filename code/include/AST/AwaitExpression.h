#pragma once

#include "AST/Expression.h"

#include <memory>

namespace rust_compiler::ast {

class AwaitExpression : public ExpressionWithoutBlock {
  std::shared_ptr<Expression> lhs;

public:
  AwaitExpression(Location loc, std::shared_ptr<Expression> lhs)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::AwaitExpression),
        lhs(lhs) {}

  std::shared_ptr<Expression> getBody() const;

  bool containsBreakExpression() override;
  size_t getTokens() override;
};

} // namespace rust_compiler::ast
