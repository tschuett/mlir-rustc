#pragma once

#include "AST/AST.h"
#include "AST/EnumItem.h"

#include <vector>

namespace rust_compiler::ast {

class EnumItems : public Node {
  bool trailingcomma;
  std::vector<EnumItem> items;

public:
  EnumItems(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast
