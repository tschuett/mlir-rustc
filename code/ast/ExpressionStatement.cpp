#include "AST/ExpressionStatement.h"

namespace rust_compiler::ast {

size_t ExpressionStatement::getTokens() { return 1 + expr->getTokens(); }

} // namespace rust_compiler::ast
