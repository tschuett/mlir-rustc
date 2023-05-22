#pragma once

#include "AST/AST.h"
#include "AST/Item.h"
#include "AST/MacroItem.h"
#include "AST/Statement.h"
#include "Location.h"
#include "AST/VisItem.h"

#include <memory>
#include <span>
#include <vector>

namespace rust_compiler::ast {

class ItemDeclaration : public Statement {
  std::vector<OuterAttribute> outerAttributes;
  std::shared_ptr<Item> visItem;
  std::shared_ptr<Item> macroItem;

public:
  ItemDeclaration(Location loc)
      : Statement(loc, StatementKind::ItemDeclaration) {}

  void setOuterAttributes(std::span<OuterAttribute> outer) {
    outerAttributes = {outer.begin(), outer.end()};
  }
  void setVisItem(std::shared_ptr<Item> vis) { visItem = vis; }
  void setMacroItem(std::shared_ptr<Item> mac) { macroItem = mac; }

  bool hasVisItem() const { return (bool)visItem; }
  bool hasMacroItem() const { return (bool)macroItem; }

  std::shared_ptr<VisItem> getVisItem() const {
    return std::static_pointer_cast<VisItem>(visItem);
  }
};

} // namespace rust_compiler::ast
