#pragma once

#include "AST/Types/QualifiedPathType.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypeNoBounds.h"
#include "Basic/Mutability.h"

#include <memory>

namespace rust_compiler::ast::types {

using namespace rust_compiler::basic;

class RawPointerType : public TypeNoBounds {
  bool mut;
  bool con;
  std::shared_ptr<ast::types::TypeExpression> type;

public:
  RawPointerType(Location loc)
      : TypeNoBounds(loc, TypeNoBoundsKind::RawPointerType) {}

  bool isMut() const { return mut; }
  bool isConst() const { return con; }

  void setMut();
  void setConst();

  Mutability getMutability() const {
    return mut ? Mutability::Mut : Mutability::Imm;
  }

  void setType(std::shared_ptr<ast::types::TypeExpression> t) { type = t; }
  std::shared_ptr<ast::types::TypeExpression> getType() const { return type; }
};

} // namespace rust_compiler::ast::types
