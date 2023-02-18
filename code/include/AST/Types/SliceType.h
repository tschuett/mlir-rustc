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
};

} // namespace rust_compiler::ast::types
