#pragma once

#include "AST/Types/TypeNoBounds.h"

namespace rust_compiler::ast::types {

class InferredType : public TypeNoBounds {
public:
  InferredType(Location loc)
      : TypeNoBounds(loc, TypeNoBoundsKind::InferredType) {}
};

} // namespace rust_compiler::ast::types
