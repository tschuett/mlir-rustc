#pragma once

#include "AST/AST.h"

#include <vector>

namespace rust_compiler::ast::types {

enum class TypeParamBoundKind { Lifetime, TraitBound };

class TypeParamBound : public Node {
  TypeParamBoundKind kind;

public:
  TypeParamBound(TypeParamBoundKind kind, Location loc)
      : Node(loc), kind(kind) {}
};

} // namespace rust_compiler::ast::types
