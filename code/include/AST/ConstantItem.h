#pragma once

#include "AST/Expression.h"
#include "AST/GenericParams.h"
#include "AST/Types/TypeExpression.h"
#include "AST/VisItem.h"
#include "AST/WhereClause.h"
#include "Location.h"

#include <optional>
#include <string>
#include <string_view>

namespace rust_compiler::ast {

class ConstantItem : public VisItem {
  std::string identifier;
  std::optional<std::shared_ptr<Expression>> init;
  std::optional<std::shared_ptr<types::TypeExpression>> type;

public:
  ConstantItem(Location loc, std::optional<Visibility> vis)
      : VisItem(loc, VisItemKind::ConstantItem, vis) {}

  void setIdentifier(std::string_view);
  void setType(std::shared_ptr<ast::types::TypeExpression>);
  void setInit(std::shared_ptr<Expression>);
};

} // namespace rust_compiler::ast
