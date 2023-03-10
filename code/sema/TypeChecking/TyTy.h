#pragma once

#include "Basic/Ids.h"

namespace rust_compiler::sema::type_checking::TyTy {

// https://rustc-dev-guide.rust-lang.org/type-inference.html#inference-variables
// https://doc.rust-lang.org/nightly/nightly-rustc/rustc_type_ir/sty/enum.TyKind.html

enum TypeKind { Int, Uint };

enum IntKind { I8, I16, I32, I64, I128 };

enum UintKind { U8, U16, U32, U64, U128 };

enum FloatKind { F32, F64 };

class BaseType {};

class IntType : public BaseType {};

class UintType : public BaseType {
public:
  UintType(basic::NodeId, UintKind);
};

} // namespace rust_compiler::sema::type_checking::TyTy
