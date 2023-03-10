#pragma once

#include "AST/Types/TypeExpression.h"

namespace rust_compiler::ast::types {

enum class TypeNoBoundsKind {
  ParenthesizedType,
  ImplTraitType,
  ImplTraitTypeOneBound,
  TraitObjectTypeOneBound,
  TypePath,
  TupleType,
  NeverType,
  RawPointerType,
  ReferenceType,
  ArrayType,
  SliceType,
  InferredType,
  QualifiedPathInType,
  BareFunctionType,
  MacroInvocation
};

class TypeNoBounds : public TypeExpression {
  TypeNoBoundsKind kind;

public:
  TypeNoBounds(Location loc, TypeNoBoundsKind kind)
      : TypeExpression(loc, TypeExpressionKind::TypeNoBounds), kind(kind) {}

  TypeNoBoundsKind getKind() const { return kind; }
};

} // namespace rust_compiler::ast::types
