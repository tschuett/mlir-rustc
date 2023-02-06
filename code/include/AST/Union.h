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
  Union(const adt::CanonicalPath &path, Location loc)
    : VisItem(path, loc, VisItemKind::Union), fields(loc) {}
};

} // namespace rust_compiler::ast
