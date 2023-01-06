#pragma once

#include "AST/AST.h"
#include "AST/Expression.h"
#include "AST/Statement.h"

#include <span>
#include <vector>
#include <memory>

namespace rust_compiler::ast {

class BlockExpression : public ExpressionWithBlock {

  std::vector<std::shared_ptr<Statement>> stmts;

public:
  BlockExpression(Location loc) : ExpressionWithBlock(loc) {}

  std::span<std::shared_ptr<Statement>> getExpressions();

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
