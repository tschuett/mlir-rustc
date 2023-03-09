#pragma once

#include "AST/AST.h"
#include "AST/Expression.h"
#include "AST/Statements.h"

#include <memory>
#include <span>
#include <vector>

namespace rust_compiler::ast {

class BlockExpression : public ExpressionWithBlock {
  Statements stmts;

public:
  BlockExpression(Location loc)
      : ExpressionWithBlock(loc, ExpressionWithBlockKind::BlockExpression),
        stmts(loc) {}

  void setStatements(const Statements &_stmts) { stmts = _stmts; }

  Statements getExpressions() { return stmts; }
};

} // namespace rust_compiler::ast
