#pragma once

#include "AST/Types/ForLifetimes.h"
#include "AST/Types/TypeParamBound.h"
#include "AST/Types/TypePath.h"

#include <vector>

namespace rust_compiler::ast::types {

class TraitBound : public TypeParamBound {
  bool hasQuestionMark = false;
  std::optional<ForLifetimes> forLifetime;
  std::shared_ptr<ast::types::TypePath> typePath;
  bool hasParenthesis = false;

public:
  TraitBound(Location loc)
      : TypeParamBound(TypeParamBoundKind::TraitBound, loc) {}

  void setHasParenthesis() { hasParenthesis = true; }
  void setHasQuestionMark() { hasQuestionMark = true; }

  void setTypePath(std::shared_ptr<ast::types::TypePath> p) { typePath = p; }
  void setForLifetimes(const ast::types::ForLifetimes &forL) {
    forLifetime = forL;
  }
};

} // namespace rust_compiler::ast::types
