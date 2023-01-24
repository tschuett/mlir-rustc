#include "AST/InfiniteLoopExpression.h"

namespace rust_compiler::ast {

std::shared_ptr<BlockExpression> InfiniteLoopExpression::getBody() const {
  return body;
}

} // namespace rust_compiler::ast
