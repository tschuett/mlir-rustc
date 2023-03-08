#pragma once

#include "AST/OperatorExpression.h"

namespace rust_compiler::ast {

class BorrowExpression final : public OperatorExpression {
  bool isMut = false;
  std::shared_ptr<Expression> expr;

public:
  BorrowExpression(Location loc)
      : OperatorExpression(loc, OperatorExpressionKind::BorrowExpression){};

  void setExpression(std::shared_ptr<Expression> expr);
  void setMut();
  void setDoubleBorrow();

  std::shared_ptr<Expression> getExpression() const;
  bool isMutable() const;
};

} // namespace rust_compiler::ast
