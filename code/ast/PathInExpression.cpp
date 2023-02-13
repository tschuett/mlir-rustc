#include "AST/PathInExpression.h"

#include "AST/PathExprSegment.h"

#include <cassert>

namespace rust_compiler::ast {

bool PathInExpression::containsBreakExpression() { return false; }

} // namespace rust_compiler::ast
