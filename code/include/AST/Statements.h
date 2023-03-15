
#pragma once

#include "AST/AST.h"
#include "AST/Expression.h"
#include "AST/Statement.h"

#include <memory>
#include <span>
#include <vector>

namespace rust_compiler::ast {

class Statements : public Node {

  std::vector<std::shared_ptr<Statement>> stmts;

  std::shared_ptr<Expression> trailing;

  bool onlySemi = false;

public:
  Statements(Location loc) : Node(loc) {}

  void addStmt(std::shared_ptr<Statement> stmt) { stmts.push_back(stmt); }
  std::vector<std::shared_ptr<Statement>> getStmts() const { return stmts; }

  void setTrailing(std::shared_ptr<Expression> trail) { trailing = trail; }

  bool hasTrailing() const { return (bool)trailing; }
  std::shared_ptr<Expression> getTrailing() const { return trailing; }

  void setOnlySemi() { onlySemi = true; };

  bool containsBreakExpression();
};

} // namespace rust_compiler::ast
