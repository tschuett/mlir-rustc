#pragma once

#include "AST/AST.h"
#include "AST/OuterAttribute.h"
#include "Location.h"

#include <memory>
#include <span>
#include <vector>

namespace rust_compiler::ast {

class VisItem;

enum class ItemKind { VisItem, MacroItem };

class Item : public Node {
  std::vector<OuterAttribute> outerAttributes;
  std::shared_ptr<VisItem> visItem;

  ItemKind kind;

public:
  explicit Item(Location loc, ItemKind kind) : Node(loc), kind(kind) {}

  void setOuterAttributes(std::span<OuterAttribute> outer);

  std::span<OuterAttribute> getOuterAttributes() ;

  ItemKind getItemKind() const { return kind; }
};

} // namespace rust_compiler::ast
