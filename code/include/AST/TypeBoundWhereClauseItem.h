#pragma once

#include "AST/Types/ForLifetimes.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypeParamBounds.h"
#include "AST/WhereClauseItem.h"

#include <memory>
#include <optional>

namespace rust_compiler::ast {

class TypeBoundWhereClauseItem : public WhereClauseItem {
  std::optional<ast::types::ForLifetimes> forLifetimes;
  std::shared_ptr<ast::types::TypeExpression> type;
  std::optional<ast::types::TypeParamBounds> bounds;

public:
  TypeBoundWhereClauseItem(Location loc)
      : WhereClauseItem(loc, WhereClauseItemKind::TypeBoundWherClauseItem) {}

  void setForLifetimes(const ast::types::ForLifetimes &forL) {
    forLifetimes = forL;
  }

  void setType(std::shared_ptr<ast::types::TypeExpression> t) { type = t; }

  void setBounds(ast::types::TypeParamBounds b) { bounds = b; }

  std::shared_ptr<ast::types::TypeExpression> getType() const { return type; }

  bool hasTypeParamBounds() const { return bounds.has_value(); }

  ast::types::TypeParamBounds getBounds() const { return *bounds; }
};

} // namespace rust_compiler::ast
