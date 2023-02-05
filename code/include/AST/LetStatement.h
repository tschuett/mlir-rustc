#pragma once

#include "AST/Expression.h"
#include "AST/LetStatementParam.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Statement.h"
#include "AST/Types/Types.h"

#include <mlir/IR/Location.h>
#include <span>

namespace rust_compiler::ast {

class LetStatement final : public Statement {
  // VariableDeclaration var;
  std::shared_ptr<ast::patterns::PatternNoTopAlt> pat;
  std::shared_ptr<ast::types::Type> type;
  std::shared_ptr<ast::Expression> expr;

  std::vector<LetStatementParam> var;
  bool filledVars = false;

public:
  LetStatement(Location loc) : Statement(loc, StatementKind::LetStatement){};

  void setPattern(std::shared_ptr<ast::patterns::PatternNoTopAlt> pat);
  void setType(std::shared_ptr<ast::types::Type> type);
  void setExpression(std::shared_ptr<ast::Expression> expr);

  std::span<LetStatementParam> getVarDecls();

  bool containsBreakExpression() override;

  size_t getTokens() override;

  std::shared_ptr<ast::patterns::PatternNoTopAlt> getPattern();

private:
};

} // namespace rust_compiler::ast
