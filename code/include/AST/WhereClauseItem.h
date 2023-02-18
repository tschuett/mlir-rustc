#pragma once

#include "AST/AST.h"

namespace rust_compiler::ast {

enum class WhereClauseItemKind {
  LifetimeWhereClauseItem,
  TypeBoundWherClauseItem
};

class WhereClauseItem : public Node {
  WhereClauseItemKind kind;

public:
  WhereClauseItem(Location loc, WhereClauseItemKind kind)
      : Node(loc), kind(kind) {}

  WhereClauseItemKind getKind() const { return kind; }
};

} // namespace rust_compiler::ast
