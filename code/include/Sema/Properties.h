#pragma once

#include <string>

namespace rust_compiler::sema {

enum class PropertyKind {
  ADD,
  SUBTRACT,
  MULTIPLY,
  DIVIDE,
  REMAINDER,
  BITAND,
  BITOR,
  BITXOR,
  SHL,
  SHR,

  NEGATION,
  NOT,

  ADD_ASSIGN,
  SUB_ASSIGN,
  MUL_ASSIGN,
  DIV_ASSIGN,
  REM_ASSIGN,
  BITAND_ASSIGN,
  BITOR_ASSIGN,
  BITXOR_ASSIGN,
  SHL_ASSIGN,
  SHR_ASSIGN,

  DEREF,
  DEREF_MUT,

  // https://github.com/rust-lang/rust/blob/master/library/core/src/ops/index.rs
  INDEX,
  INDEX_MUT,

  // https://github.com/rust-lang/rust/blob/master/library/core/src/ops/range.rs
  RANGE_FULL,
  RANGE,
  RANGE_FROM,
  RANGE_TO,
  RANGE_INCLUSIVE,
  RANGE_TO_INCLUSIVE,

  // https://github.com/rust-lang/rust/blob/master/library/core/src/marker.rs
  PHANTOM_DATA,

  // functions
  FN,
  FN_MUT,
  FN_ONCE,
  FN_ONCE_OUTPUT,

  // markers
  COPY,
  CLONE,
  SIZED,

  // https://github.com/rust-lang/rust/commit/afbecc0f68c4dcfc4878ba5bcb1ac942544a1bdc
  // https://github.com/rust-lang/rust/blob/master/library/core/src/ptr/const_ptr.rs
  SLICE_ALLOC,
  SLICE_U8_ALLOC,
  STR_ALLOC,
  ARRAY,
  BOOL,
  CHAR,
  F32,
  F64,
  I8,
  I16,
  I32,
  I64,
  I128,
  ISIZE,
  U8,
  U16,
  U32,
  U64,
  U128,
  USIZE,
  CONST_PTR,
  CONST_SLICE_PTR,
  MUT_PTR,
  MUT_SLICE_PTR,
  SLICE_U8,
  SLICE,
  STR,
  F32_RUNTIME,
  F64_RUNTIME,

  // delimiter
  UNKNOWN,
};

std::string Property2String(PropertyKind);

} // namespace rust_compiler::sema
