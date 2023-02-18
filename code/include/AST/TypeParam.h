#pragma once

#include "AST/AST.h"
#include "AST/Types/TypeParamBounds.h"

#include <optional>
#include <string>

namespace rust_compiler::ast {

class TypeParam : public Node {
  std::string identifier;
  std::optional<types::TypeParamBounds> bounds;
  std::optional<std::shared_ptr<types::TypeExpression>> type;

public:
  TypeParam(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast
