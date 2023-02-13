#pragma once

#include "AST/VisItem.h"
#include "Location.h"

#include <optional>

namespace rust_compiler::ast {

class Union : public VisItem {
  std::string identifier;
  std::optional<GenericParams> genericParams;
  std::optional<WhereClause> whereClause;
  StructFields fields;

public:
  Union(Location loc, std::optional<Visibility> vis)
    : VisItem(loc, VisItemKind::Union, vis), fields(loc) {}
};

} // namespace rust_compiler::ast
