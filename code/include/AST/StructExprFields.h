#pragma once

#include "AST/StructBase.h"
#include "AST/StructExprField.h"

#include <optional>
#include <variant>

namespace rust_compiler::ast {

class StructExprFields : public Node {
  std::vector<StructExprField> fields;

  std::optional<StructBase> base;

  bool trailingComma = false;

public:
  StructExprFields(Location loc) : Node(loc) {}

  void addField(const StructExprField &f) { fields.push_back(f); }
  void setBase(const StructBase &b) { base = b; }
  void setTrailingcomma() { trailingComma = true; }
};

} // namespace rust_compiler::ast
