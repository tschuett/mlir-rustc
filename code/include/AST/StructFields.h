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

  void addStructField(const StructField &sf) { fields.push_back(sf); }
  void setTrailingComma() { trailingComma = true; }

  std::vector<StructField> getFields() const { return fields; }
};

} // namespace rust_compiler::ast
