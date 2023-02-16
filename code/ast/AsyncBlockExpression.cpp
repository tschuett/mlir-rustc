#include "AST/AsyncBlockExpression.h"

namespace rust_compiler::ast {

void AsyncBlockExpression::setMove() { move = true; }

void AsyncBlockExpression::setBlock(std::shared_ptr<Expression> e) { block = e; }

} // namespace rust_compiler::ast
