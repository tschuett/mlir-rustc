#include "AST/LiteralExpression.h"

#include <memory>

using namespace rust_compiler::ast::types;

namespace rust_compiler::ast {

bool LiteralExpression::containsBreakExpression() { return false; }

} // namespace rust_compiler::ast
