#pragma once

#include "AST/AST.h"
#include "AST/TupleField.h"

#include <optional>

namespace rust_compiler::ast {

class TupleFields : public Node {
  bool trailingComma;
  std::vector<TupleField> fields;

public:
  TupleFields(Location loc) : Node(loc) {}

  bool hasTrailingComma() const { return trailingComma; }

  void addField(const TupleField &t) { fields.push_back(t); }

  std::vector<TupleField> getFields() const { return fields; }
};

} // namespace rust_compiler::ast
