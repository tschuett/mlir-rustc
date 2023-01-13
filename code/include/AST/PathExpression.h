#pragma once

#include "AST/Expression.h"

namespace rust_compiler::ast {

enum class PathExpressionKind { PathInExpression, QualifiedPathInExpression };

class PathExpression : public ExpressionWithoutBlock {
  PathExpressionKind kind;

public:
  PathExpression(rust_compiler::Location loc, PathExpressionKind kind)
      : ExpressionWithoutBlock(loc, ExpressionWithoutBlockKind::PathExpression),
        kind(kind) {}

  PathExpressionKind getPathExpressionKind() const { return kind; }
  //  size_t getTokens() override;
};

} // namespace rust_compiler::ast
