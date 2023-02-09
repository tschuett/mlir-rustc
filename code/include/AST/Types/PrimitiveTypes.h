#pragma once

// https://doc.rust-lang.org/reference/types.html

#include "AST/Types/Types.h"

#include <optional>
#include <string>

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
  Isize,
  Never,
  F32,
  F64,
  Unit
};

class PrimitiveType : public Type {
  PrimitiveTypeKind kind;

public:
  PrimitiveType(Location loc, PrimitiveTypeKind kind)
      : Type(loc, TypeKind::PrimitiveType), kind(kind) {}

  PrimitiveTypeKind getKind() const { return kind; }

   size_t getTokens() override;
};

std::optional<std::string> PrimitiveType2String(PrimitiveTypeKind);

std::optional<PrimitiveTypeKind> isPrimitiveType(std::string_view identifier);

} // namespace rust_compiler::ast::types



// The () type, also called “unit”.
