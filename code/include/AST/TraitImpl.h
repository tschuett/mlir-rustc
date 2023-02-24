#pragma once

#include "AST/AssociatedItem.h"
#include "AST/GenericParams.h"
#include "AST/Implementation.h"
#include "AST/InnerAttribute.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypePath.h"
#include "AST/Types/Types.h"
#include "AST/WhereClause.h"

#include <optional>

namespace rust_compiler::ast {

class TraitImpl : public Implementation {
  bool unsafe = false;
  bool notKw = false;
  std::optional<GenericParams> genericParams;
  std::shared_ptr<types::TypeExpression> type;
  std::optional<WhereClause> whereClause;
  std::shared_ptr<ast::types::TypePath> typePath;

  std::vector<InnerAttribute> innerAttributes;
  std::vector<AssociatedItem> associatedItems;

public:
  TraitImpl(Location loc, std::optional<Visibility> vis)
      : Implementation(ImplementationKind::TraitImpl, loc, vis) {}

  std::span<AssociatedItem> getAssociatedItems() const;

  void setUnsafe() { unsafe = true; }
  void setGenericParams(const GenericParams &gp) { genericParams = gp; }
  void setNot() { notKw = true; }
  void setTypePath(std::shared_ptr<ast::types::TypePath> tp) { typePath = tp; }
  void setType(std::shared_ptr<ast::types::TypeExpression> te) { type = te; }
  void setInnerAttributes(std::span<InnerAttribute> i) {
    innerAttributes = {i.begin(), i.end()};
  }
  void setWhereClause(const WhereClause &wc) { whereClause = wc; }
  void addAssociatedItem(const ast::AssociatedItem &as) {
    associatedItems.push_back(as);
  }
};

} // namespace rust_compiler::ast
