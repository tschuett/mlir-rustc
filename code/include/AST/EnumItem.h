#pragma once

#include "AST/AST.h"
#include "AST/OuterAttribute.h"
#include "EnumItemDiscriminant.h"
#include "EnumItemStruct.h"
#include "EnumItemTuple.h"

#include <span>
#include <string>
#include <variant>
#include <vector>

namespace rust_compiler::ast {

class EnumItem : public Node {
  std::vector<OuterAttribute> outerAttributes;
  std::optional<Visibility> visibility;
  std::string identifier;
  std::optional<
      std::variant<EnumItemTuple, EnumItemStruct, EnumItemDiscriminant>>
      item;

public:
  EnumItem(Location loc) : Node(loc) {}

  void setOuterAttributes(std::span<OuterAttribute> attr) {
    outerAttributes = {attr.begin(), attr.end()};
  }

  void setVisibility(const Visibility &vis) { visibility = vis; }

  void setIdentifier(std::string_view i) { identifier = i; }

  void setEnumItemTuple(const EnumItemTuple &tu) { item = tu; }
  void setEnumItemStruct(const EnumItemStruct &st) { item = st; }
  void setEnumItemDiscriminant(const EnumItemDiscriminant &dis) { item = dis; }

};

} // namespace rust_compiler::ast
