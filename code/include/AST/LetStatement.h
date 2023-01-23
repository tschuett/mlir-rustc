#pragma once

#include "AST/Expression.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Statement.h"
#include "AST/Types/Types.h"
#include "AST/VariableDeclaration.h"

#include <mlir/IR/Location.h>

namespace rust_compiler::ast {

class LetStatement : public Statement {
  // VariableDeclaration var;
  std::shared_ptr<ast::patterns::PatternNoTopAlt> pat;
  std::shared_ptr<ast::types::Type> type;
  std::shared_ptr<ast::Expression> expr;

  std::vector<VariableDeclaration> var;
  bool filledVars = false;

public:
  LetStatement(Location loc) : Statement(loc, StatementKind::LetStatement){};

  void setPattern(std::shared_ptr<ast::patterns::PatternNoTopAlt> pat);
  void setType(std::shared_ptr<ast::types::Type> type);
  void setExpression(std::shared_ptr<ast::Expression> expr);

  std::vector<VariableDeclaration> getVarDecls();

  size_t getTokens() override;

private:
  std::shared_ptr<ast::patterns::PatternNoTopAlt> getPattern();
};

} // namespace rust_compiler::ast
