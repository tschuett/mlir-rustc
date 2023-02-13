#pragma once

#include "AST/AST.h"
#include "AST/OuterAttribute.h"
#include "AST/TupleFields.h"

#include <optional>

namespace rust_compiler::ast {

class EnumItemTuple : public Node {
  std::optional<TupleFields> fields;

public:
  EnumItemTuple(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast
