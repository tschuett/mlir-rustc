#pragma once

#include "AST/Expression.h"

#include <memory>

namespace rust_compiler::ast {

class DelimTokenTree;

enum class AttrInputKind { DelimTokenTree, Expression };

class AttrInput : public Node {
  std::shared_ptr<DelimTokenTree> tree;
  std::shared_ptr<Expression> expr;
  AttrInputKind kind;

public:
  AttrInput(Location loc) : Node(loc) {}

  AttrInputKind getKind() const;

  void setTokenTree(std::shared_ptr<DelimTokenTree> _tree) {
    tree = _tree;
    kind = AttrInputKind::DelimTokenTree;
  }
  void setExpression(std::shared_ptr<Expression> _expr) {
    expr = _expr;
    kind = AttrInputKind::Expression;
  }
};

} // namespace rust_compiler::ast
