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
  CompoundAssignmentExpression(rust_compiler::Location loc,
                               CompoundAssignmentExpressionKind kind,
                               std::shared_ptr<Expression> left,
                               std::shared_ptr<Expression> right)
      : OperatorExpression(
            loc, OperatorExpressionKind::CompoundAssignmentExpression),
        kind(kind), left(left), right(right) {}

  CompoundAssignmentExpressionKind getKind() const { return kind; }

  std::shared_ptr<Expression> getRHS() const { return right; };

  std::shared_ptr<Expression> getLHS() const { return left; };

  bool containsBreakExpression() override;

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
