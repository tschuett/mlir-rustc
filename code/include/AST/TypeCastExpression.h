#pragma once

#include "AST/AST.h"
#include "AST/OperatorExpression.h"
#include "AST/Types/TypeExpression.h"

namespace rust_compiler::ast {

class TypeCastExpression : public OperatorExpression {
  std::shared_ptr<Expression> expr;
  std::shared_ptr<types::TypeExpression> type;

public:
  TypeCastExpression(Location loc)
      : OperatorExpression(loc, OperatorExpressionKind::TypeCastExpression) {}

  void setLeft(std::shared_ptr<Expression> l) { expr = l; }
  void setType(std::shared_ptr<types::TypeExpression> no) { type = no; }

  std::shared_ptr<Expression> getLeft() const { return expr; }
  std::shared_ptr<types::TypeExpression> getRight() const { return type; }
};

} // namespace rust_compiler::ast
