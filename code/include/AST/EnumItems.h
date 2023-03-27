#pragma once

#include "AST/AST.h"
#include "AST/EnumItem.h"

#include <vector>

namespace rust_compiler::ast {

class EnumItems : public Node {
  bool trailingComma;
  std::vector<EnumItem> items;

public:
  EnumItems(Location loc) : Node(loc) {}

  bool hasTrailingComma() const { return trailingComma; }

  void addItem(const EnumItem &it) { items.push_back(it); }

  void setTrailingComma() { trailingComma = true; }

  std::vector<EnumItem> getItems() const { return items; }
};

} // namespace rust_compiler::ast
