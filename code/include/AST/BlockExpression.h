#pragma once

#include "AST/AST.h"
#include "AST/Expression.h"
#include "AST/Statements.h"

#include <memory>
#include <span>
#include <vector>

namespace rust_compiler::ast {

class BlockExpression : public ExpressionWithBlock {

  std::shared_ptr<Statements> stmts;

public:
  BlockExpression(Location loc)
      : ExpressionWithBlock(loc, ExpressionWithBlockKind::BlockExpression) {}

  void setStatements(std::shared_ptr<Statements> stmts);

  std::shared_ptr<Statements> getExpressions();
  bool containsBreakExpression()  override;

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
