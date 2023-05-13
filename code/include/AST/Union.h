#pragma once

#include "AST/GenericParams.h"
#include "AST/StructFields.h"
#include "AST/VisItem.h"
#include "AST/WhereClause.h"
#include "Lexer/Identifier.h"
#include "Location.h"

#include <optional>

namespace rust_compiler::ast {

class Union : public VisItem {
  Identifier identifier;
  std::optional<GenericParams> genericParams;
  std::optional<WhereClause> whereClause;
  StructFields fields;

public:
  Union(Location loc, std::optional<Visibility> vis)
      : VisItem(loc, VisItemKind::Union, vis), fields(loc) {}

  void setGenericParams(const GenericParams &gp) { genericParams = gp; }
  void setWhereClause(const WhereClause &wc) { whereClause = wc; }
  void setStructfields(const StructFields &sf) { fields = sf; }
  void setIdentifier(const Identifier &i) { identifier = i; }

  bool hasGenericParams() const { return genericParams.has_value(); }
  bool hasWhereClause() const { return whereClause.has_value(); }

  GenericParams getGenericParams() const { return *genericParams; }
  WhereClause getWhereClause() const { return *whereClause; }
  StructFields getStructFields() const { return fields; }
  lexer::Identifier getIdentifier() const { return identifier; }
};

} // namespace rust_compiler::ast
