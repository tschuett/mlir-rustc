#pragma once

#include "AST/StructBase.h"
#include "AST/StructExprField.h"

#include <optional>
#include <variant>

namespace rust_compiler::ast {

class StructExprFields : public Node {
  std::vector<StructExprField> fields;

  std::optional<StructBase> base;

  bool trailingComma;

public:
  StructExprFields(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast
