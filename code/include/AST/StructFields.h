#pragma once

#include "AST/AST.h"
#include "AST/StructField.h"

#include <optional>

namespace rust_compiler::ast {

class StructFields : public Node {
  bool trailingComma;
  std::vector<StructField> fields;

public:
  StructFields(Location loc) : Node(loc) {}

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
