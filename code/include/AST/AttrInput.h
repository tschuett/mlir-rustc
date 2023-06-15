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
  std::vector<std::shared_ptr<MetaItemInner>> items;

public:
  AttrInput(Location loc) : Node(loc) {}

  AttrInputKind getKind() const;

//  AttrInput(const AttrInput &other);
//
//  AttrInput &operator=(const AttrInput &other);

//  // default move semantics
//  AttrInput(AttrInput &&other) = default;
//  AttrInput &operator=(AttrInput &&other) = default;

  void setTokenTree(std::shared_ptr<DelimTokenTree> _tree) {
    tree = _tree;
    kind = AttrInputKind::DelimTokenTree;
  }

  void setExpression(std::shared_ptr<Expression> _expr) {
    expr = _expr;
    kind = AttrInputKind::Expression;
  }

  void setMetaItems(std::vector<std::shared_ptr<MetaItemInner>> items) {
    this->items = items;
    kind = AttrInputKind::MetaItems;
  }

  void parseToMetaItem();

  //std::shared_ptr<AttrInput> clone();

  const std::vector<std::shared_ptr<MetaItemInner>>& getMetaItems() const {
    return items;
  }
};

} // namespace rust_compiler::ast
