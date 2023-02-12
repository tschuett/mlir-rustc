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

/// https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/type.TyKind.html

enum class TypeKind {
  Bool,
  Uint,
  Int,
  Float,
  Char,
  Str,
  Never,
  Tuple,
  Array,
  Slice,
  Struct,
  Enum,
  Union,
  Function,
  Closure,
  Ref,
  RawPointer,
  FunctionPointer,
  FIXME
};

class Type : public Node {
  TypeKind kind;

public:
  Type(Location loc, TypeKind kind) : Node(loc), kind(kind) {}

  TypeKind getKind() const { return kind; }

  //  size_t getTokens() override;
};

} // namespace rust_compiler::ast::types
