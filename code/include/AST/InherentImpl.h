#pragma once

#include "AST/AssociatedItem.h"
#include "AST/GenericParams.h"
#include "AST/Implementation.h"
#include "AST/InnerAttribute.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/Types.h"
#include "AST/WhereClause.h"

#include <optional>
#include <vector>
#include <span>
#include <memory>

namespace rust_compiler::ast {

class InherentImpl : public Implementation {
  std::optional<GenericParams> genericParams;
  std::shared_ptr<types::TypeExpression> type;
  std::optional<WhereClause> whereClause;

  std::vector<InnerAttribute> innerAttributes;
  std::vector<AssociatedItem> associatedItems;

public:
  InherentImpl(Location loc,
               std::optional<Visibility> vis)
      : Implementation(ImplementationKind::InherentImpl, loc, vis) {}

  std::span<AssociatedItem> getAssociatedItems() const;

  void setInnerAttributes(std::span<InnerAttribute> i) {
    innerAttributes = {i.begin(), i.end()};
  }
  void setGenericParams(const GenericParams &p) { genericParams = p; }
  void setWhereClause(const WhereClause &wc) { whereClause = wc; }
  void addAssociatedItem(const ast::AssociatedItem &as) {
    associatedItems.push_back(as);
  }
  void setType(std::shared_ptr<ast::types::TypeExpression> te) { type = te; }
};

} // namespace rust_compiler::ast
