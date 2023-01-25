#include "AST/AwaitExpression.h"

namespace rust_compiler::ast {

std::shared_ptr<Expression> AwaitExpression::getBody() const { return body; }

} // namespace rust_compiler::ast
