#pragma once

#include "AST/Types/TraitBound.h"
#include "AST/Types/TypeNoBounds.h"

namespace rust_compiler::ast::types {

class ImplTraitTypeOneBound : public TypeNoBounds {
  std::shared_ptr<ast::types::TypeParamBound> bound;

public:
  ImplTraitTypeOneBound(Location loc)
      : TypeNoBounds(loc, TypeNoBoundsKind::ImplTraitTypeOneBound) {}

  void setBound(std::shared_ptr<ast::types::TypeParamBound> tb) { bound = tb; }

  std::shared_ptr<ast::types::TypeParamBound> getBound() const { return bound; }
};

} // namespace rust_compiler::ast::types
