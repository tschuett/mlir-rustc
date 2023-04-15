#pragma once

#include "AST/AssociatedItem.h"
#include "AST/GenericParams.h"
#include "AST/Implementation.h"
#include "AST/InnerAttribute.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/Types.h"
#include "AST/WhereClause.h"

#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace rust_compiler::ast {

class InherentImpl : public Implementation {
  std::optional<GenericParams> genericParams;
  std::shared_ptr<types::TypeExpression> type;
  std::optional<WhereClause> whereClause;

  std::vector<InnerAttribute> innerAttributes;
  std::vector<AssociatedItem> associatedItems;

public:
  InherentImpl(Location loc, std::optional<Visibility> vis)
      : Implementation(ImplementationKind::InherentImpl, loc, vis) {}

  void setInnerAttributes(std::span<InnerAttribute> i) {
    innerAttributes = {i.begin(), i.end()};
  }
  void setGenericParams(const GenericParams &p) { genericParams = p; }
  void setWhereClause(const WhereClause &wc) { whereClause = wc; }
  void addAssociatedItem(const ast::AssociatedItem &as) {
    associatedItems.push_back(as);
  }
  void setType(std::shared_ptr<ast::types::TypeExpression> te) { type = te; }

  bool hasGenericParams() const { return genericParams.has_value(); }
  GenericParams getGenericParams() const { return *genericParams; }

  bool hasWhereClause() const { return whereClause.has_value(); }
  WhereClause getWhereClause() const { return *whereClause; }

  std::shared_ptr<types::TypeExpression> getType() const { return type; }

  std::vector<AssociatedItem> getAssociatedItems() const {
    return associatedItems;
  }
};

} // namespace rust_compiler::ast
