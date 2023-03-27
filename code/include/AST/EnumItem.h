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

enum class EnumItemKind { Tuple, Struct, Discriminant };

class EnumItem : public Node {
  std::vector<OuterAttribute> outerAttributes;
  std::optional<Visibility> visibility;
  std::string identifier;
  std::optional<
      std::variant<EnumItemTuple, EnumItemStruct, EnumItemDiscriminant>>
      item;

  EnumItemKind kind;

public:
  EnumItem(Location loc) : Node(loc) {}

  void setOuterAttributes(std::span<OuterAttribute> attr) {
    outerAttributes = {attr.begin(), attr.end()};
  }

  void setVisibility(const Visibility &vis) { visibility = vis; }

  void setIdentifier(std::string_view i) { identifier = i; }

  void setEnumItemTuple(const EnumItemTuple &tu) {
    item = tu;
    kind = EnumItemKind::Tuple;
  }
  void setEnumItemStruct(const EnumItemStruct &st) {
    item = st;
    kind = EnumItemKind::Struct;
  }
  void setEnumItemDiscriminant(const EnumItemDiscriminant &dis) {
    item = dis;
    kind = EnumItemKind::Discriminant;
  }

  EnumItemKind getKind() const { return kind; }

  EnumItemDiscriminant getDiscriminant() const {
    return std::get<EnumItemDiscriminant>(*item);
  }
  EnumItemStruct getStruct() const { return std::get<EnumItemStruct>(*item); }
  EnumItemTuple getTuple() const { return std::get<EnumItemTuple>(*item); }

  std::string getName() const { return identifier; }
};

} // namespace rust_compiler::ast
