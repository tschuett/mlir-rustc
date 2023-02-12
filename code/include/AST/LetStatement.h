#pragma once

#include "AST/Expression.h"
#include "AST/BlockExpression.h"
#include "AST/LetStatementParam.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Statement.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/Types.h"
#include "AST/OuterAttribute.h"

#include <mlir/IR/Location.h>
#include <span>

namespace rust_compiler::ast {

class LetStatement final : public Statement {
  // VariableDeclaration var;
  std::vector<OuterAttribute> outerAttributes;
  std::shared_ptr<ast::patterns::PatternNoTopAlt> pat;
  std::optional<std::shared_ptr<ast::types::TypeExpression>> type;
  std::optional<std::shared_ptr<ast::Expression>> expr;
  std::optional<std::shared_ptr<ast::BlockExpression>> elseExpr;

  std::vector<LetStatementParam> var;
  bool filledVars = false;

public:
  LetStatement(Location loc) : Statement(loc, StatementKind::LetStatement){};

  void setPattern(std::shared_ptr<ast::patterns::PatternNoTopAlt> pat);
  void setType(std::shared_ptr<ast::types::TypeExpression> type);
  void setExpression(std::shared_ptr<ast::Expression> expr);

  //std::span<LetStatementParam> getVarDecls();

  bool containsBreakExpression() override;

  size_t getTokens() override;

  std::shared_ptr<ast::patterns::PatternNoTopAlt> getPattern();

private:
};

} // namespace rust_compiler::ast
