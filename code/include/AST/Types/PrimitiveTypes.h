#pragma once

// https://doc.rust-lang.org/reference/types.html

#include "AST/Types/Types.h"

namespace rust_compiler::ast::types {

enum class PrimitiveTypeKind {
  Boolean,
  U8,
  U16,
  U32,
  U64,
  U128,
  I8,
  I16,
  I32,
  I64,
  I128,
  Char,
  Str,
  Binary32,
  Binary64,
  Usize,
  Isize
};

class PrimitiveType : public Type {};

} // namespace rust_compiler::ast::types
