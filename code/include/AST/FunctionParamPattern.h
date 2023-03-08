#pragma once

#include "AST/AST.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Types/TypeExpression.h"

#include <memory>
#include <optional>

namespace rust_compiler::ast {

class FunctionParamPattern : public Node {
  std::optional<std::shared_ptr<ast::types::TypeExpression>> type;
  std::shared_ptr<ast::patterns::PatternNoTopAlt> name;

public:
  FunctionParamPattern(Location loc) : Node(loc) {}

  void setName(std::shared_ptr<ast::patterns::PatternNoTopAlt> name);

  void setType(std::shared_ptr<ast::types::TypeExpression> type);

  std::shared_ptr<ast::patterns::PatternNoTopAlt> getPattern() const {
    return name;
  }

  std::shared_ptr<ast::types::TypeExpression> getType() const { return *type; }

  bool hasType() const { return type.has_value(); }
};

} // namespace rust_compiler::ast
