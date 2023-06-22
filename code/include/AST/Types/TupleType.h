#pragma once

#include "AST/Types/TypeNoBounds.h"

#include <memory>
#include <vector>

namespace rust_compiler::ast::types {

class TupleType : public TypeNoBounds {

  std::vector<std::shared_ptr<ast::types::TypeExpression>> types;

public:
  TupleType(Location loc) : TypeNoBounds(loc, TypeNoBoundsKind::TupleType) {}

  void addType(std::shared_ptr<ast::types::TypeExpression> t) {
    types.push_back(t);
  }

  std::vector<std::shared_ptr<ast::types::TypeExpression>> getTypes() const {
    return types;
  }
};

} // namespace rust_compiler::ast::types
