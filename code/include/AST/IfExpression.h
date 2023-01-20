#pragma once

#include "AST/Expression.h"
#include "Location.h"

namespace rust_compiler::ast {

class IfExpression final : public ExpressionWithBlock {
  std::shared_ptr<ast::Expression> condition;
  std::shared_ptr<ast::Expression> block;
  std::shared_ptr<ast::Expression> trailing;

public:
  IfExpression(Location loc)
      : ExpressionWithBlock(loc, ExpressionWithBlockKind::IfExpression) {}

  void setCondition(std::shared_ptr<ast::Expression> condition);

  std::shared_ptr<ast::Expression> getCondition() const;

  void setBlock(std::shared_ptr<ast::Expression> block);

  std::shared_ptr<ast::Expression> getBlock() const;

  void setTrailing(std::shared_ptr<ast::Expression> block);

  std::shared_ptr<ast::Expression> getTrailing() const;

  bool hasTrailing() const;

  size_t getTokens() override;

  std::shared_ptr<ast::types::Type> getType() override;
};

} // namespace rust_compiler::ast
