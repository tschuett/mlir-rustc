#pragma once

#include "AST/Types/TypeNoBounds.h"
#include "AST/Types/TypeParamBounds.h"

namespace rust_compiler::ast::types {

class ImplTraitTypeOneBound : public TypeNoBounds {
  std::shared_ptr<ast::types::TypeParamBound> bound;

public:
  ImplTraitTypeOneBound(Location loc)
      : TypeNoBounds(loc, TypeNoBoundsKind::ImplTraitTypeOneBound) {}

  void setBound(std::shared_ptr<ast::types::TypeParamBound> b) { bound = b; }
};

} // namespace rust_compiler::ast::types
