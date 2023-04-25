#pragma once

#include "AST/AST.h"
#include "AST/Types/TypeParamBounds.h"
#include "Lexer/Identifier.h"

#include <optional>
#include <string>
#include <string_view>

namespace rust_compiler::ast {

class TypeParam : public Node {
  lexer::Identifier identifier;
  std::optional<types::TypeParamBounds> bounds;
  std::optional<std::shared_ptr<types::TypeExpression>> type;

public:
  TypeParam(Location loc) : Node(loc) {}

  void setIdentifier(const lexer::Identifier &id) { identifier = id; }
  void setBounds(types::TypeParamBounds b) { bounds = b; }
  void setType(std::shared_ptr<ast::types::TypeExpression> t) { type = t; }

  bool hasType() const { return type.has_value(); }
  bool hasTypeParamBounds() const { return bounds.has_value(); }

  std::shared_ptr<ast::types::TypeExpression> getType() const { return *type; }
  types::TypeParamBounds getBounds() const { return *bounds; }
  lexer::Identifier getIdentifier() const { return identifier; }
};

} // namespace rust_compiler::ast
