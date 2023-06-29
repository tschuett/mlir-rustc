#pragma once

#include "AST/AST.h"
#include "AST/Types/Types.h"

#include <memory>
#include <vector>

namespace rust_compiler::ast::types {

class TypePathFnInputs final : public Node {
  std::vector<std::shared_ptr<types::TypeExpression>> types;
  bool trailingcomma;

public:
  TypePathFnInputs(Location loc) : Node(loc) {}

  void setTrailingComma() { trailingcomma = true; }
  void addType(std::shared_ptr<types::TypeExpression> t) { types.push_back(t); }

  std::vector<std::shared_ptr<types::TypeExpression>> getTypes() const {
    return types;
  }
};

} // namespace rust_compiler::ast::types
