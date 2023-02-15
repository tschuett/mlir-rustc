#pragma once

#include "AST/Expression.h"
#include "AST/Patterns/Pattern.h"
#include "AST/Scrutinee.h"
#include "Location.h"

namespace rust_compiler::ast {

class IfLetExpression final : public ExpressionWithBlock {
  std::shared_ptr<Scrutinee> scrutinee;
  std::shared_ptr<ast::patterns::Pattern> pattern;
  std::shared_ptr<ast::Expression> block;
  std::shared_ptr<ast::Expression> trailing;

public:
  IfLetExpression(Location loc)
      : ExpressionWithBlock(loc, ExpressionWithBlockKind::IfLetExpression) {}

  void setPattern(std::shared_ptr<ast::patterns::Pattern> pattern);

  void setScrutinee(std::shared_ptr<ast::Scrutinee> scrutinee);

  void setBlock(std::shared_ptr<ast::Expression> block);

  void setTrailing(std::shared_ptr<ast::Expression> block);
};

} // namespace rust_compiler::ast
