#pragma once

#include "AST/AssociatedItem.h"
#include "AST/Expression.h"
#include "AST/GenericParams.h"
#include "AST/InnerAttribute.h"
#include "AST/Types/Types.h"
#include "AST/Types/TypeParamBounds.h"
#include "AST/VisItem.h"
#include "AST/WhereClause.h"
#include "Location.h"

#include <optional>
#include <string>

namespace rust_compiler::ast {

class Trait : public VisItem {
  bool unsafe;
  std::string identifier;
  std::optional<GenericParams> genericParams;
  std::optional<types::TypeParamBounds> typeParamBounds;
  std::optional<WhereClause> whereClause;

  std::vector<InnerAttribute> innerAttributes;
  std::vector<AssociatedItem> associatedItem;

public:
  Trait(const adt::CanonicalPath &path, Location loc)
      : VisItem(path, loc, VisItemKind::Trait) {}
};

} // namespace rust_compiler::ast
