#pragma once

#include "AST/AST.h"
#include "AST/Types/TypePathSegment.h"
#include "AST/Types/Types.h"

#include <vector>

namespace rust_compiler::ast::types {

class TypePath final : public Node {
  bool trailingDoubleColon;
  std::vector<TypePathSegment> typePathSegments;

public:
  TypePath(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast::types
