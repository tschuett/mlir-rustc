#include "AST/Function.h"

namespace rust_compiler::ast {

std::shared_ptr<BlockExpression> Function::getBody() { return body; }

mlir::Location Function::getLocation() const { return location; }

FunctionSignature Function::getSignature() const { return signature; }

FunctionQualifiers Function::getFunctionQualifiers() const { return qualifiers; };

} // namespace rust_compiler::ast
