#pragma once

#include "AST/AST.h"
#include "AST/OuterAttribute.h"
#include "EnumItemTuple.h"
#include "EnumItemStruct.h"
#include "EnumItemDiscriminant.h"

#include <vector>
#include <variant>
#include <string>

namespace rust_compiler::ast {

class EnumItem : public Node {
  std::vector<OuterAttribute> outerAttributes;
  // std::optional<Visibility>
  std::string identifier;
  std::optional<
      std::variant<EnumItemTuple, EnumItemStruct, EnumItemDiscriminant>>
      item;

public:
  EnumItem(Location loc) : Node(loc) {}

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
