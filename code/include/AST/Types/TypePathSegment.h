#pragma once

#include "AST/AST.h"
#include "AST/Types/PathIdentSegment.h"
#include "AST/GenericArgs.h"
#include "AST/Types/TypePathFn.h"

#include <vector>
#include <optional>

namespace rust_compiler::ast::types {

class TypePathSegment final : public Node {
  PathIdentSegment pathIdentSegment;
  bool doubleColon;
  std::optional<std::variant<GenericArgs, TypePathFn>> tail;

public:
  TypePathSegment(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast::types
