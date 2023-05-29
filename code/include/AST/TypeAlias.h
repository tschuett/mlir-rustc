#pragma once

#include "AST/GenericParams.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypeParamBounds.h"
#include "AST/Types/Types.h"
#include "AST/VisItem.h"
#include "AST/WhereClause.h"
#include "Lexer/Identifier.h"
#include "Location.h"

#include <optional>

namespace rust_compiler::ast {

using namespace rust_compiler::lexer;

class TypeAlias : public VisItem {
  Identifier identifier;
  std::optional<GenericParams> genericParams;
  std::optional<types::TypeParamBounds> typeParamBounds;
  std::optional<WhereClause> whereClause;
  std::optional<WhereClause> whereClauseType;

  std::optional<std::shared_ptr<types::TypeExpression>> type;

public:
  TypeAlias(Location loc, std::optional<Visibility> vis)
      : VisItem(loc, VisItemKind::TypeAlias, vis) {}

  void setIdentifier(const Identifier &s) { identifier = s; }
  void setGenericParams(const GenericParams &gp) { genericParams = gp; }
  void setParamBounds(const types::TypeParamBounds &tpb) {
    typeParamBounds = tpb;
  }
  void setWhereClause(const WhereClause &wc) { whereClause = wc; }
  void setTypeWhereClause(const WhereClause &wc) { whereClauseType = wc; }
  void setType(std::shared_ptr<types::TypeExpression> expr) { type = expr; }

  Identifier getIdentifier() const { return identifier; }

  bool hasGenericParams() const { return genericParams.has_value(); }
  GenericParams getGenericParams() const { return *genericParams; }

  bool hasWhereClause() const { return whereClause.has_value(); }
  WhereClause getWhereClause() const { return *whereClause; }

  bool hasType() const { return type.has_value(); }
  std::shared_ptr<types::TypeExpression> getType() const { return *type; }
};

} // namespace rust_compiler::ast
