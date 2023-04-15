#pragma once

#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypeParamBounds.h"

namespace rust_compiler::ast::types {

class TraitObjectType : public TypeExpression {
  bool dyn = false;
  TypeParamBounds bounds;

public:
  TraitObjectType(Location loc)
      : TypeExpression(loc, TypeExpressionKind::TraitObjectType), bounds(loc) {}

  void setBounds(const TypeParamBounds &b) { bounds = b; }

  bool hasDyn() const { return dyn; }
};

} // namespace rust_compiler::ast::types
