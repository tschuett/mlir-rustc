#include "AST/Function.h"

namespace rust_compiler::ast {

std::shared_ptr<BlockExpression> Function::getBody() { return body; }

mlir::Location Function::getLocation() { return location; }

FunctionSignature Function::getSignature() { return signature; }

} // namespace rust_compiler::ast
