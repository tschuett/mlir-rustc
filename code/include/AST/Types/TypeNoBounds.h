#pragma once

#include "AST/Types/TypeExpression.h"

namespace rust_compiler::ast::types {

enum class TypeNoBoundsKind { ParenthesisedType, ImplTraitTypeOneBound };

class TypeNoBounds : public TypeExpression {
  TypeNoBoundsKind kind;

  TypeNoBounds(Location loc, TypeNoBoundsKind kind)
      : TypeExpression(loc, TypeExpressionKind::TypeNoBounds), kind(kind) {}
};

} // namespace rust_compiler::ast::types
