#pragma once

#include "AST/EnumItems.h"
#include "AST/GenericParams.h"
#include "AST/VisItem.h"
#include "AST/WhereClause.h"

#include <optional>
#include <string>
#include <string_view>

namespace rust_compiler::ast {

class Enumeration : public VisItem {
  std::string identifier;
  std::optional<GenericParams> genericParams;
  std::optional<WhereClause> whereClause;
  std::optional<EnumItems> enumItems;

public:
  Enumeration(Location loc, std::optional<Visibility> vis)
      : VisItem(loc, VisItemKind::Enumeration, vis) {}

  void setGenericParams(const GenericParams &gp) { genericParams = gp; }

  void setWhereClause(const WhereClause &wc) { whereClause = wc; }

  void setIdentifier(std::string_view id) { identifier = id; }

  void setItems(const EnumItems &ei) { enumItems = ei; }

  bool hasGenericParams() const { return genericParams.has_value(); }

  GenericParams getGenericParams() const { return *genericParams; }

  bool hasWhereClause() const { return whereClause.has_value(); }

  WhereClause getWhereClause() const { return *whereClause; }

  bool hasEnumItems() const { return enumItems.has_value(); }
  EnumItems getEnumItems() const { return *enumItems; }

  std::string getName() const { return identifier; }
};

} // namespace rust_compiler::ast
