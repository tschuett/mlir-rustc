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

// enum class EnumItemKind { Tuple, Struct, Discriminant };

class EnumItem : public Node {
  std::vector<OuterAttribute> outerAttributes;
  std::optional<Visibility> visibility;
  Identifier identifier;
  std::optional<std::variant<EnumItemTuple, EnumItemStruct>> item;
  std::optional<EnumItemDiscriminant> discriminant;

  // EnumItemKind kind;

public:
  EnumItem(Location loc) : Node(loc) {}

  void setOuterAttributes(std::span<OuterAttribute> attr) {
    outerAttributes = {attr.begin(), attr.end()};
  }

  void setVisibility(const Visibility &vis) { visibility = vis; }

  void setIdentifier(const Identifier &i) { identifier = i; }

  void setEnumItemTuple(const EnumItemTuple &tu) { item = tu; }
  void setEnumItemStruct(const EnumItemStruct &st) { item = st; }
  void setEnumItemDiscriminant(const EnumItemDiscriminant &dis) {
    discriminant = dis;
  }

  bool hasTuple() const {
    return item.has_value() and std::holds_alternative<EnumItemTuple>(*item);
  }
  bool hasStruct() const {
    return item.has_value() and std::holds_alternative<EnumItemStruct>(*item);
  }
  bool hasDiscriminant() const { return discriminant.has_value(); }

  EnumItemDiscriminant getDiscriminant() const { return *discriminant; }
  EnumItemStruct getStruct() const { return std::get<EnumItemStruct>(*item); }
  EnumItemTuple getTuple() const { return std::get<EnumItemTuple>(*item); }

  Identifier getName() const { return identifier; }
};

} // namespace rust_compiler::ast
