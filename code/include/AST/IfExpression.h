#pragma once

#include "AST/Expression.h"
#include "Location.h"

namespace rust_compiler::ast {

class IfExpression : public ExpressionWithBlock {
  std::shared_ptr<ast::Expression> condition;
  std::shared_ptr<ast::Expression> block;
  std::shared_ptr<ast::Expression> trailing;

public:
  IfExpression(Location loc)
      : ExpressionWithBlock(loc, ExpressionWithBlockKind::IfExpression) {}

  void setCondition(std::shared_ptr<ast::Expression> condition);

  void setBlock(std::shared_ptr<ast::Expression> block);

  void setTrailing(std::shared_ptr<ast::Expression> block);

  size_t getTokens() override;

  std::shared_ptr<ast::types::Type> getType() override;
};

} // namespace rust_compiler::ast
