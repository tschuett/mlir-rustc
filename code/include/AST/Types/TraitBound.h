#pragma once

#include "AST/Types/TypeParamBound.h"
#include "AST/Types/ForLifetimes.h"

#include <vector>

namespace rust_compiler::ast::types {

class TraitBound : public TypeParamBound {
  bool questionMark;
  std::optional<ForLifetimes> forLifetime;
  TypePath typePath;
public:
  TraitBound(Location loc)
      : TypeParamBound(TypeParamBoundKind::TraitBound, loc) {}
};

} // namespace rust_compiler::ast::types
