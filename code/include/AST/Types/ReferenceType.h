#pragma once

#include "AST/Types/Lifetime.h"
#include "AST/Types/TypeNoBounds.h"
#include "AST/Types/Types.h"
#include "Basic/Mutability.h"

#include <memory>
#include <optional>

namespace rust_compiler::ast::types {

class ReferenceType : public TypeNoBounds {
  bool mut;
  std::shared_ptr<ast::types::TypeExpression> noBounds;

  // FIXME Lifetime
  std::optional<ast::Lifetime> lifetime;

public:
  ReferenceType(Location loc)
      : TypeNoBounds(loc, TypeNoBoundsKind::ReferenceType) {}

  void setMut() { mut = true; }
  void setType(std::shared_ptr<ast::types::TypeExpression> t) { noBounds = t; }

  void setLifetime(ast::Lifetime l) { lifetime = l; }

  std::shared_ptr<ast::types::TypeExpression> getReferencedType() const {
    return noBounds;
  }

  bool isMutable() const { return mut; }

  basic::Mutability getMut() const {
    return mut ? basic::Mutability::Mut : basic::Mutability::Imm;
  }
};

} // namespace rust_compiler::ast::types
