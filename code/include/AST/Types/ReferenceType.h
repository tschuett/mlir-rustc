#pragma once

#include "AST/Types/TypeNoBounds.h"
#include "AST/Types/Types.h"

#include <memory>

namespace rust_compiler::ast::types {

class ReferenceType : public TypeNoBounds {
  bool mut;
  std::shared_ptr<ast::types::TypeNoBounds> noBounds;

  // FIXME Lifetime
public:
  ReferenceType(Location loc)
      : TypeNoBounds(loc, TypeNoBoundsKind::ReferenceType) {}

  void setMut() { mut = true; }
  void setType(std::shared_ptr<ast::types::TypeNoBounds> t) { noBounds = t; }
};

} // namespace rust_compiler::ast::types
