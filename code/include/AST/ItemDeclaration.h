#pragma once

#include "AST/AST.h"
#include "AST/Item.h"
#include "AST/MacroItem.h"
#include "AST/Statement.h"
#include "Location.h"

#include <memory>
#include <span>
#include <vector>

namespace rust_compiler::ast {

class ItemDeclaration : public Statement {
  std::vector<OuterAttribute> outerAttributes;
  std::shared_ptr<VisItem> visItem;
  std::shared_ptr<MacroItem> macroItem;

public:
  ItemDeclaration(Location loc)
      : Statement(loc, StatementKind::ItemDeclaration) {}

  void setOuterAttributes(std::span<OuterAttribute> outer) {
    outerAttributes = {outer.begin(), outer.end()};
  }
  void setVisItem(std::shared_ptr<VisItem> vis) { visItem = vis; }
  void setMacroItem(std::shared_ptr<MacroItem> mac) { macroItem = mac; }
};

} // namespace rust_compiler::ast
