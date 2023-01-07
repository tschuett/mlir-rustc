#pragma once

#include "AST/Expression.h"

namespace rust_compiler::ast {

class PathExpression : public Expression {};

class PathInExpression : public PathExpression {};

class QualifiedPathInExpression : public PathExpression {};

} // namespace rust_compiler::ast
