#pragma once

#include "AST/PathExpression.h"

namespace rust_compiler::ast {

class QualifiedPathInExpression : public PathExpression {
public:
  size_t getTokens() override;
};

} // namespace rust_compiler::ast
