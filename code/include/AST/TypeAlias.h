#pragma once

#include "AST/VisItem.h"
#include "Location.h"
#include "AST/WhereClause.h"
#include "AST/GenericParams.h"
#include "AST/Types/Types.h"
#include "AST/Types/TypeParamBounds.h"

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
  TypeAlias(const adt::CanonicalPath &path, Location loc)
      : VisItem(path, loc, VisItemKind::TypeAlias) {}
};

} // namespace rust_compiler::ast
