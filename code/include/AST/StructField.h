#pragma once

#include "AST/AST.h"
#include "AST/OuterAttribute.h"
#include "AST/StructField.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/Types.h"
#include "AST/Visiblity.h"

#include <optional>
#include <string>
#include <vector>

namespace rust_compiler::ast {

using namespace rust_compiler::lexer;

class StructField : public Node {
  std::vector<OuterAttribute> outerAttributes;
  std::optional<Visibility> visibility;
  Identifier identifier;
  std::shared_ptr<ast::types::TypeExpression> type;

public:
  StructField(Location loc) : Node(loc) {}

  void setOuterAttributes(const std::vector<OuterAttribute> &o) {
    outerAttributes = o;
  }
  void setVisibility(const Visibility &vis) { visibility = vis; }
  void setIdentifier(const Identifier& id) { identifier = id; }
  void setType(std::shared_ptr<ast::types::TypeExpression> t) { type = t; }

  std::optional<Visibility> getVisibility() const { return visibility; }
  std::shared_ptr<ast::types::TypeExpression> getType() const { return type; }
};

} // namespace rust_compiler::ast
