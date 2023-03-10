#pragma once

#include "AST/AST.h"

namespace rust_compiler::ast::types {

enum class TypeExpressionKind { TypeNoBounds, ImplTraitType, TraitObjectType };

class TypeExpression : public Node {
  TypeExpressionKind kind;

public:
  TypeExpression(Location loc, TypeExpressionKind kind)
      : Node(loc), kind(kind) {}

  TypeExpressionKind getKind() const { return kind; }

  // size_t getTokens() override;
};

} // namespace rust_compiler::ast::types
