#pragma once

#include "AST/Expression.h"
#include "AST/Statement.h"
#include "AST/Types/Types.h"
#include "AST/VariableDeclaration.h"
#include "AST/Patterns/PatternNoTopAlt.h"

#include <mlir/IR/Location.h>

namespace rust_compiler::ast {

class LetStatement : public Statement {
  // VariableDeclaration var;
  std::shared_ptr<ast::patterns::PatternNoTopAlt> pat;
  std::shared_ptr<ast::types::Type> type;
  std::shared_ptr<ast::Expression> expr;

public:
  LetStatement(Location loc) : Statement(loc, StatementKind::LetStatement){};

  void setPattern(std::shared_ptr<ast::patterns::PatternNoTopAlt> pat);
  void setType(std::shared_ptr<ast::types::Type> type);
  void setExpression(std::shared_ptr<ast::Expression> expr);

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
