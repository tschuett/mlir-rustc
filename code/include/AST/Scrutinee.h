#pragma once

#include "AST/AST.h"
#include "AST/Expression.h"

#include <memory>

namespace rust_compiler::ast {

class Scrutinee : public Node {
  std::shared_ptr<ast::Expression> expr;

public:
  Scrutinee(Location loc) : Node(loc){};

  void setExpression(std::shared_ptr<ast::Expression> _expr) { expr = _expr; }

  size_t getTokens() override {
    assert(expr);
    return expr->getTokens();
  }
};

} // namespace rust_compiler::ast

// FIXME:
