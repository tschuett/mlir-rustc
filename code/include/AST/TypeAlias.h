#pragma once

#include "AST/GenericParams.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypeParamBounds.h"
#include "AST/Types/Types.h"
#include "AST/VisItem.h"
#include "AST/WhereClause.h"
#include "Location.h"

#include <optional>

namespace rust_compiler::ast {

class TypeAlias : public VisItem {
  std::string identifier;
  std::optional<GenericParams> genericParams;
  std::optional<types::TypeParamBounds> typeParamBounds;
  std::optional<WhereClause> whereClause;
  std::optional<WhereClause> whereClauseType;

  std::optional<std::shared_ptr<types::TypeExpression>> type;

public:
  TypeAlias(Location loc, std::optional<Visibility> vis)
      : VisItem(loc, VisItemKind::TypeAlias, vis) {}

  void setIdentifier(std::string_view s) { identifier = s; }
  void setGenericParams(const GenericParams &gp) { genericParams = gp; }
  void setParamBounds(const types::TypeParamBounds &tpb) {
    typeParamBounds = tpb;
  }
  void setWhereClause(const WhereClause &wc) { whereClause = wc; }
  void setTypeWhereClause(const WhereClause &wc) { whereClauseType = wc; }
  void setType(std::shared_ptr<types::TypeExpression> expr) { type = expr; }
};

} // namespace rust_compiler::ast
