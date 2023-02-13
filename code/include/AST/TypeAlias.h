#pragma once

#include "AST/GenericParams.h"
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
  std::optional<WhereClause> whereClause2;

  std::optional<std::shared_ptr<types::Type>> type;

public:
  TypeAlias(Location loc, std::optional<Visibility> vis)
      : VisItem(loc, VisItemKind::TypeAlias, vis) {}
};

} // namespace rust_compiler::ast
