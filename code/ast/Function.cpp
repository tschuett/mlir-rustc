#include "AST/Function.h"

#include "AST/BlockExpression.h"

namespace rust_compiler::ast {

std::shared_ptr<BlockExpression> Function::getBody() { return body; }

Location Function::getLocation() const { return location; }

FunctionSignature Function::getSignature() const { return signature; }

FunctionQualifiers Function::getFunctionQualifiers() const {
  return qualifiers;
};

void Function::setSignature(FunctionSignature _nature) { signature = _nature; }

void Function::setBody(std::shared_ptr<BlockExpression> _body) { body = _body; }

size_t Function::getTokens() {
  assert(false);
  return 0;
};

} // namespace rust_compiler::ast
