#pragma once

#include "AST/Expression.h"
#include "AST/OperatorExpression.h"

#include <memory>

namespace rust_compiler::ast {

enum class CompoundAssignmentExpressionKind {
  Add,
  Sub,
  Mul,
  Div,
  Rem,
  And,
  Or,
  Xor,
  Shl,
  Shr
};

class CompoundAssignmentExpression final : public OperatorExpression {
  CompoundAssignmentExpressionKind kind;
  std::shared_ptr<Expression> left;
  std::shared_ptr<Expression> right;

public:
  CompoundAssignmentExpression(rust_compiler::Location loc)
      : OperatorExpression(
            loc, OperatorExpressionKind::CompoundAssignmentExpression) {}

  CompoundAssignmentExpressionKind getKind() const { return kind; }

  std::shared_ptr<Expression> getRHS() const { return right; };

  std::shared_ptr<Expression> getLHS() const { return left; };

  void setLhs(std::shared_ptr<Expression> e) { left = e; }
  void setRhs(std::shared_ptr<Expression> r) { right = r; }

  void setKind(CompoundAssignmentExpressionKind e) { kind = e; }
};

} // namespace rust_compiler::ast
