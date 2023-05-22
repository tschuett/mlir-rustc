#pragma once

#include "AST/Types/TypeNoBounds.h"

#include <memory>

namespace rust_compiler::ast::types {

class SliceType : public TypeNoBounds {
  std::shared_ptr<ast::types::TypeExpression> type;

public:
  SliceType(Location loc) : TypeNoBounds(loc, TypeNoBoundsKind::SliceType) {}

  void setType(std::shared_ptr<ast::types::TypeExpression> _type) {
    type = _type;
  }

  std::shared_ptr<ast::types::TypeExpression> getType() const { return type; }
};

} // namespace rust_compiler::ast::types
