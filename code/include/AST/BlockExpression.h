#pragma once

#include "AST/Expression.h"
#include "AST/InnerAttribute.h"
#include "AST/Statements.h"

#include <memory>
#include <span>
#include <vector>

namespace rust_compiler::ast {

class BlockExpression : public ExpressionWithBlock {
  Statements stmts;
  std::vector<InnerAttribute> inner;

public:
  BlockExpression(Location loc)
      : ExpressionWithBlock(loc, ExpressionWithBlockKind::BlockExpression),
        stmts(loc) {}

  void setInnerAttributes(std::span<InnerAttribute> i) {
    inner = {i.begin(), i.end()};
  };
  void setStatements(const Statements &_stmts) { stmts = _stmts; }

  Statements getExpressions() const { return stmts; }
};

} // namespace rust_compiler::ast
