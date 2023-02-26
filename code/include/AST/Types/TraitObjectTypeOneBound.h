#pragma once

#include "AST/Types/TypeNoBounds.h"
#include "AST/Types/TypeParamBounds.h"

namespace rust_compiler::ast::types {

class TraitObjectTypeOneBound : public TypeNoBounds {
  std::shared_ptr<ast::types::TypeParamBound> bound;
  bool dyn = false;
public:
  TraitObjectTypeOneBound(Location loc)
      : TypeNoBounds(loc, TypeNoBoundsKind::TraitObjectTypeOneBound) {}

  void setBound(std::shared_ptr<ast::types::TypeParamBound> b) { bound = b; }

  void setDyn() { dyn = true;}
};

} // namespace rust_compiler::ast::types
