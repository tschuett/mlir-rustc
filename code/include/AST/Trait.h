#pragma once

#include "AST/AssociatedItem.h"
#include "AST/Expression.h"
#include "AST/GenericParams.h"
#include "AST/InnerAttribute.h"
#include "AST/Types/TypeParamBounds.h"
#include "AST/Types/Types.h"
#include "AST/VisItem.h"
#include "AST/WhereClause.h"
#include "Location.h"

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace rust_compiler::ast {

class Trait : public VisItem {
  bool unsafe;
  std::string identifier;
  std::optional<GenericParams> genericParams;
  std::optional<types::TypeParamBounds> typeParamBounds;
  std::optional<WhereClause> whereClause;

  std::vector<InnerAttribute> innerAttributes;
  std::vector<std::shared_ptr<AssociatedItem>> associatedItem;

public:
  Trait(Location loc, std::optional<Visibility> vis)
      : VisItem(loc, VisItemKind::Trait, vis) {}

  std::span<std::shared_ptr<AssociatedItem>> getAssociatedItems() const;
};

} // namespace rust_compiler::ast
