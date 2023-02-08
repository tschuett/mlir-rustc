#pragma once

#include "AST/Expression.h"
#include "AST/VisItem.h"
#include "Location.h"
#include "AST/WhereClause.h"
#include "AST/GenericParams.h"
#include "AST/Types/TypeExpression.h"

#include <optional>
#include <string>

namespace rust_compiler::ast {

class ConstantItem : public VisItem {
  std::string identifier;
  std::optional<std::shared_ptr<Expression>> init;

  std::optional<std::shared_ptr<types::TypeExpression>> type;

public:
  ConstantItem(const adt::CanonicalPath &path, Location loc)
      : VisItem(path, loc, VisItemKind::ConstantItem) {}
};

} // namespace rust_compiler::ast
