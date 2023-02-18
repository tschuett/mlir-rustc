#pragma once

#include "AST/Types/QualifiedPathType.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypeNoBounds.h"

#include <vector>

namespace rust_compiler::ast::types {

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

  void setType(std::shared_ptr<ast::types::TypeExpression> t) { type = t; }
};

} // namespace rust_compiler::ast::types
