#pragma once

#include "AST/DelimTokenTree.h"
#include "AST/Expression.h"

#include <memory>

namespace rust_compiler::ast {

enum class AttrInputKind { DelimTokenTree, Expression };

class AttrInput : public Node {
  DelimTokenTree tree;
  std::shared_ptr<Expression> expr;
  AttrInputKind kind;

public:
  AttrInput(Location loc) : Node(loc), tree(loc) {}

  AttrInputKind getKind() const;

  void setTokenTree(const DelimTokenTree &_tree) {
    tree = _tree;
    kind = AttrInputKind::DelimTokenTree;
  }
  void setExpression(std::shared_ptr<Expression> _expr) {
    expr = _expr;
    kind = AttrInputKind::Expression;
  }
};

} // namespace rust_compiler::ast
