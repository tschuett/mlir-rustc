#pragma once

#include "AST/Expression.h"
#include "AST/Statement.h"

#include <memory>

namespace rust_compiler::ast {

enum class ExpressionStatementKind {
  ExpressionWithoutBlock,
  ExpressionWithBlock
};

class ExpressionStatement : public Statement {
  std::shared_ptr<Expression> wo;
  std::shared_ptr<Expression> with;
  bool trailingSemi = false;

  ExpressionStatementKind kind;

public:
  ExpressionStatement(Location loc)
      : Statement(loc, StatementKind::ExpressionStatement) {}

  ExpressionStatementKind getKind() const { return kind; }
  void setExprWoBlock(std::shared_ptr<Expression> w) {
    wo = w;
    kind = ExpressionStatementKind::ExpressionWithoutBlock;
  }
  void setExprWithBlock(std::shared_ptr<Expression> w) {
    with = w;
    kind = ExpressionStatementKind::ExpressionWithBlock;
  }

  void setTrailingSemi() { trailingSemi = true; }
};

} // namespace rust_compiler::ast
