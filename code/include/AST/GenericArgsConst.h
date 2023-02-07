#pragma once

#include "AST/AST.h"

namespace rust_compiler::ast {

enum class GenericArgsConstKind {
  BlockExpression,
  LiteralExpression,
  SimplePathSegment
};

class GenericArgsConst : public Node {
public:
  size_t getTokens() override;
};

} // namespace rust_compiler::ast