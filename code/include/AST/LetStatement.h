#pragma once

#include "AST/Expression.h"
#include "AST/LetStatementParam.h"
#include "AST/OuterAttribute.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Statement.h"
#include "AST/Types/TypeExpression.h"

#include <mlir/IR/Location.h>
#include <span>

namespace rust_compiler::ast {

class LetStatement final : public Statement {
  // VariableDeclaration var;
  std::vector<OuterAttribute> outerAttributes;
  std::shared_ptr<ast::patterns::PatternNoTopAlt> pat;
  std::optional<std::shared_ptr<ast::types::TypeExpression>> type;
  std::optional<std::shared_ptr<ast::Expression>> expr;
  std::optional<std::shared_ptr<ast::Expression>> elseExpr;

  std::vector<LetStatementParam> var;

public:
  LetStatement(Location loc) : Statement(loc, StatementKind::LetStatement){};

  void setOuterAttributes(std::span<OuterAttribute> out) {
    outerAttributes = {out.begin(), out.end()};
  }

  void setPattern(std::shared_ptr<ast::patterns::PatternNoTopAlt> pat);
  void setType(std::shared_ptr<ast::types::TypeExpression> type);
  void setExpression(std::shared_ptr<ast::Expression> expr);
  void setElseExpr(std::shared_ptr<ast::Expression> exp) { elseExpr = exp; }

  std::shared_ptr<ast::patterns::PatternNoTopAlt> getPattern();
};

} // namespace rust_compiler::ast
