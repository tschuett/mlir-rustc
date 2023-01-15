#pragma once

#include "AST/AST.h"

namespace rust_compiler::ast {

enum class OuterAttributeKind {
  ClippyAttribute,
};

class OuterAttribute : public Node {
  OuterAttributeKind kind;

public:
  OuterAttribute(Location loc, OuterAttributeKind kind)
      : Node(loc), kind(kind) {}

  OuterAttributeKind getOuterAttributeKind() { return kind; }

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
