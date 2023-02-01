#pragma once

#include "AST/AST.h"
#include "AST/OuterAttribute.h"

#include <vector>
#include <variant>
#include <string>

namespace rust_compiler::ast {

class EnumItem : public Node {
  std::vector<OuterAttribute> outerAttributes;
  // Visibility
  std::string identifier;
  std::optional<
      std::variant<EnumItemTuple, EnumItemStruct, EnumItemDisciminant>>
      item;

public:
  EnumItem(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast
