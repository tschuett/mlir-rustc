#pragma once

#include "AST/Expression.h"
#include "AST/MetaItemInner.h"

#include <memory>
#include <vector>


namespace rust_compiler::ast {

class AttrInputMetaItemContainer;

class DelimTokenTree;

enum class AttrInputKind { DelimTokenTree, Expression, MetaItems };

class AttrInput : public Node {
  std::shared_ptr<DelimTokenTree> tree;
  std::shared_ptr<Expression> expr;
  AttrInputKind kind;
  std::vector<std::unique_ptr<MetaItemInner>> items;

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

  void setMetaItems(std::vector<std::unique_ptr<MetaItemInner>> items) {
    this->items = std::move(items);
    kind = AttrInputKind::MetaItems;
  }

  void parseToMetaItem();
};

} // namespace rust_compiler::ast
