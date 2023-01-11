#pragma once

#include "AST/AST.h"

// https://doc.rust-lang.org/reference/types.html

namespace rust_compiler::ast::types {

// Primitive types

// sequence types

// user-defined types

// function types

// pointer types

// trait types

enum class TypeKind {
  PrimitiveType
};

class Type : public Node {
  TypeKind kind;

public:
  Type(Location loc, TypeKind kind) : Node(loc) {}

  TypeKind getKind() const { return kind; }

  size_t getTokens() override { return 1; };
};

} // namespace rust_compiler::ast::types
