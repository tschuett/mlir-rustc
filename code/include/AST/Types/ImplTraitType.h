#pragma once

#include "AST/Types/TypeNoBounds.h"
#include "AST/Types/TypeParamBounds.h"

namespace rust_compiler::ast::types {

class ImplTraitType : public TypeNoBounds {
  TypeParamBounds bounds;

public:
  ImplTraitType(Location loc)
      : TypeNoBounds(loc, TypeNoBoundsKind::ImplTraitType), bounds(loc) {}

  void setBounds(const TypeParamBounds &b) { bounds = b; }
};

} // namespace rust_compiler::ast::types
