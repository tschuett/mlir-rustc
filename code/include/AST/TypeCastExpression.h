#pragma once

#include "AST/AST.h"
#include "AST/OperatorExpression.h"
#include "AST/Types/TypeNoBounds.h"

namespace rust_compiler::ast {

class TypeCastExpression : public OperatorExpression {
  std::shared_ptr<Expression> expr;
  std::shared_ptr<types::TypeNoBounds> type;

public:
  TypeCastExpression(Location loc)
      : OperatorExpression(loc, OperatorExpressionKind::TypeCastExpression) {}
};

} // namespace rust_compiler::ast
