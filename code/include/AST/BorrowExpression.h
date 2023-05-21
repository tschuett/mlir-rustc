#pragma once

#include "AST/OperatorExpression.h"
#include "Basic/Mutability.h"

namespace rust_compiler::ast {

class BorrowExpression final : public OperatorExpression {
  bool isMut = false;
  std::shared_ptr<Expression> expr;
  bool doubleBorrow = false;

public:
  BorrowExpression(Location loc)
      : OperatorExpression(loc, OperatorExpressionKind::BorrowExpression){};

  void setExpression(std::shared_ptr<Expression> expr);
  void setMut();
  void setDoubleBorrow() { doubleBorrow = true; }

  std::shared_ptr<Expression> getExpression() const;
  bool isMutable() const;

  basic::Mutability getMutability() const {
    return isMut ? basic::Mutability::Mut : basic::Mutability::Imm;
  }
};

} // namespace rust_compiler::ast
