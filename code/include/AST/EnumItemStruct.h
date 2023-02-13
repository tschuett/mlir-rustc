#pragma once

#include "AST/AST.h"
#include "AST/OuterAttribute.h"
#include "AST/StructFields.h"

#include <optional>

namespace rust_compiler::ast {

class EnumItemStruct : public Node {
  std::optional<StructFields> fields;

public:
  EnumItemStruct(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast
