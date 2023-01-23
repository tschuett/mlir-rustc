#pragma once

#include "AST/Expression.h"
#include "AST/Statement.h"

namespace rust_compiler::ast {

class ExpressionStatement : public Statement {
  std::shared_ptr<Expression> expr;

public:
  ExpressionStatement(Location loc, std::shared_ptr<Expression> expr)
    : Statement(loc, StatementKind::ExpressionStatement), expr(expr) {}

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
