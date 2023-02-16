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

class StaticItem : public VisItem {
  bool mut;
  std::string identifier;

  std::optional<std::shared_ptr<types::TypeExpression>> type;
  std::optional<std::shared_ptr<Expression>> init;

public:
  StaticItem(Location loc, std::optional<Visibility> vis)
      : VisItem(loc, VisItemKind::StaticItem, vis) {}

  void setMut();
  void setIdentifier(std::string_view);
  void setType(std::shared_ptr<types::TypeExpression>);
  void setInit(std::shared_ptr<Expression>);
};

} // namespace rust_compiler::ast
