#pragma once

#include "AST/GenericParams.h"
#include "AST/StructFields.h"
#include "AST/VisItem.h"
#include "AST/WhereClause.h"
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

  void setGenericParams(const GenericParams &gp) { genericParams = gp; }
  void setWhereClause(const WhereClause &wc) { whereClause = wc; }
  void setStructfields(const StructFields &sf) { fields = sf; }
};

} // namespace rust_compiler::ast
