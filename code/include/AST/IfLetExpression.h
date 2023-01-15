#pragma once

#include "AST/Expression.h"
#include "AST/Scrutinee.h"
#include "Location.h"
#include "AST/Patterns/Pattern.h"

namespace rust_compiler::ast {

class IfLetExpression : public ExpressionWithBlock {

public:
  IfLetExpression(Location loc)
      : ExpressionWithBlock(loc, ExpressionWithBlockKind::IfLetExpression) {}

  void setPattern(std::shared_ptr<ast::patterns::Pattern> pattern);

  void setScrutinee(ast::Scrutinee scrutinee);

  void setBlock(std::shared_ptr<ast::Expression> block);

  void setTrailing(std::shared_ptr<ast::Expression> block);

  size_t getTokens() override;

  std::shared_ptr<ast::types::Type> getType() override;
};

} // namespace rust_compiler::ast
