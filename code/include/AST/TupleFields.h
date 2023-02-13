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
};

} // namespace rust_compiler::ast
