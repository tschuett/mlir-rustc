#pragma once

#include "AST/Expression.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypeNoBounds.h"

#include <memory>

namespace rust_compiler::ast::types {

class ArrayType : public TypeNoBounds {
  std::shared_ptr<ast::types::TypeExpression> type;
  std::shared_ptr<ast::Expression> expr;

public:
  ArrayType(Location loc) : TypeNoBounds(loc, TypeNoBoundsKind::ArrayType) {}

  void setType(std::shared_ptr<ast::types::TypeExpression> _type) {
    type = _type;
  }

  void setExpression(std::shared_ptr<ast::Expression> _expr) { expr = _expr; }

  std::shared_ptr<ast::types::TypeExpression> getType() const { return type; }
  std::shared_ptr<ast::Expression> getExpression() const { return expr; }
};

} // namespace rust_compiler::ast::types
