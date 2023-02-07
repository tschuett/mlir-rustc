#pragma once

#include "AST/AssociatedItem.h"
#include "AST/GenericParams.h"
#include "AST/Implementation.h"
#include "AST/InnerAttribute.h"
#include "AST/Types/Types.h"
#include "AST/WhereClause.h"

#include <optional>

namespace rust_compiler::ast {

class TraitImpl : public Implementation {
  std::optional<GenericParams> genericParams;
  std::shared_ptr<types::Type> type;
  std::optional<WhereClause> whereClause;

  std::vector<InnerAttribute> innerAttributes;
  std::vector<AssociatedItem> associatedItem;

public:
  TraitImpl(const adt::CanonicalPath &path, ImplementationKind kind,
               Location loc)
      : Implementation(path, ImplementationKind::TraitImpl, loc) {}
};

} // namespace rust_compiler::ast