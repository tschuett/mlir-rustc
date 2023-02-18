#pragma once

#include "AST/Types/TypeNoBounds.h"

namespace rust_compiler::ast::types {

class NeverType : public TypeNoBounds {
public:
  NeverType(Location loc) : TypeNoBounds(loc, TypeNoBoundsKind::NeverType) {}

  };

} // namespace rust_compiler::ast::types
