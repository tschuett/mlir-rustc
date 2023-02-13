#pragma once

#include "AST/Expression.h"
#include "AST/Statement.h"

namespace rust_compiler::ast {

enum class ExpressionStatementKind {
  ExpressionWithoutBlock,
  ExpressionWithBlock,
};

class ExpressionStatement : public Statement {
  std::shared_ptr<Expression> expr;

public:
  ExpressionStatement(Location loc, std::shared_ptr<Expression> expr)
      : Statement(loc, StatementKind::ExpressionStatement), expr(expr) {}

  bool containsBreakExpression() override;

  ExpressionStatementKind getKind() const;
};

} // namespace rust_compiler::ast
