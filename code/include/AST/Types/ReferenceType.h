#pragma once

#include "AST/Types/TypeNoBounds.h"
#include "AST/Types/Types.h"
#include "Basic/Mutability.h"

#include <memory>

namespace rust_compiler::ast::types {

class ReferenceType : public TypeNoBounds {
  bool mut;
  std::shared_ptr<ast::types::TypeExpression> noBounds;

  // FIXME Lifetime
public:
  ReferenceType(Location loc)
      : TypeNoBounds(loc, TypeNoBoundsKind::ReferenceType) {}

  void setMut() { mut = true; }
  void setType(std::shared_ptr<ast::types::TypeExpression> t) { noBounds = t; }

  std::shared_ptr<ast::types::TypeExpression> getReferencedType() const {
    return noBounds;
  }

  basic::Mutability getMut() const {
    return mut ? basic::Mutability::Mut : basic::Mutability::Imm;
  }
};

} // namespace rust_compiler::ast::types
