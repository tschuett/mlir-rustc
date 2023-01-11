#pragma once

#include "AST/AST.h"
#include "AST/Statement.h"
#include "Location.h"

namespace rust_compiler::ast {

enum class ItemKind {
  Function,
  Module,
  InnerAttribute,
  UseDeclaration,
  ClippyAttribute
};

class Item : public Node {
  ItemKind kind;

public:
  explicit Item(rust_compiler::Location location, ItemKind kind)
      : Node(location), kind(kind) {}

  ItemKind getKind() const { return kind; }
};

} // namespace rust_compiler::ast
