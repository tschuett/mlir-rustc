#pragma once

#include "AST/Types/TypeNoBounds.h"

#include <memory>
#include <vector>

namespace rust_compiler::ast::types {

class ParenthesizedType : public TypeNoBounds {

  std::shared_ptr<ast::types::TypeExpression> type;

public:
  ParenthesizedType(Location loc)
      : TypeNoBounds(loc, TypeNoBoundsKind::ParenthesizedType) {}

  void setType(std::shared_ptr<ast::types::TypeExpression> t) { type = t; }

  std::shared_ptr<ast::types::TypeExpression> getType() const { return type; }
};

} // namespace rust_compiler::ast::types
