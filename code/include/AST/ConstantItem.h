#pragma once

#include "AST/Expression.h"
#include "AST/GenericParams.h"
#include "AST/Types/TypeExpression.h"
#include "AST/VisItem.h"
#include "AST/WhereClause.h"
#include "Lexer/Identifier.h"
#include "Location.h"

#include <optional>
#include <string>
#include <string_view>

namespace rust_compiler::ast {

using namespace rust_compiler::lexer;

class ConstantItem : public VisItem {
  Identifier identifier;
  std::optional<std::shared_ptr<Expression>> init;
  std::shared_ptr<types::TypeExpression> type;

  bool underscore = false;

public:
  ConstantItem(Location loc, std::optional<Visibility> vis)
      : VisItem(loc, VisItemKind::ConstantItem, vis) {}

  void setIdentifier(const Identifier &);
  void setUnderscore() { underscore = true; }
  void setType(std::shared_ptr<ast::types::TypeExpression>);
  void setInit(std::shared_ptr<Expression>);

  Identifier getName() const {
    if (underscore)
      return Identifier("_");
    return identifier;
  }
  std::shared_ptr<types::TypeExpression> getType() const { return type; }
  bool hasInit() const { return init.has_value(); }
  std::shared_ptr<Expression> getInit() const { return *init; }
};

} // namespace rust_compiler::ast
