#pragma once

#include "AST/Expression.h"

namespace rust_compiler::ast {

class ReturnExpression : public ExpressionWithoutBlock {
  std::shared_ptr<ast::Expression> expr;

public:
  ReturnExpression(Location loc, std::shared_ptr<ast::Expression> expr)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::ReturnExpression),
        expr(expr) {}

  ReturnExpression(Location loc)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::ReturnExpression) {}

  std::shared_ptr<ast::Expression> getExpression();

  size_t getTokens() override;

  std::shared_ptr<ast::types::Type> getType() override;
};

} // namespace rust_compiler::ast
