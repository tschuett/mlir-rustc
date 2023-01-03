#include "AST/Function.h"

namespace rust_compiler::ast {

std::shared_ptr<BlockExpression> Function::getBody() { return body; }

mlir::Location Function::getLocation() const { return location; }

FunctionSignature Function::getSignature() const { return signature; }

FunctionQualifiers Function::getFunctionQualifiers() const {
  return qualifiers;
};

size_t Function::getTokens() {
  assert(false);
  return 0;
};

} // namespace rust_compiler::ast
