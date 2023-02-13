#pragma once

#include "AST/AssociatedItem.h"
#include "AST/GenericParams.h"
#include "AST/Implementation.h"
#include "AST/InnerAttribute.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/Types.h"
#include "AST/WhereClause.h"

#include <optional>

namespace rust_compiler::ast {

class TraitImpl : public Implementation {
  std::optional<GenericParams> genericParams;
  std::shared_ptr<types::TypeExpression> type;
  std::optional<WhereClause> whereClause;

  std::vector<InnerAttribute> innerAttributes;
  std::vector<AssociatedItem> associatedItem;

public:
  TraitImpl(ImplementationKind kind, Location loc,
            std::optional<Visibility> vis)
      : Implementation(ImplementationKind::TraitImpl, loc, vis) {}

  std::span<AssociatedItem> getAssociatedItems() const;
};

} // namespace rust_compiler::ast
