#include "AST/Function.h"
#include "AST/Visiblity.h"

#include "AST/BlockExpression.h"
#include "llvm/Support/raw_ostream.h"

namespace rust_compiler::ast {

std::shared_ptr<BlockExpression> Function::getBody() { return body; }

Location Function::getLocation() const { return location; }

FunctionSignature Function::getSignature() const { return signature; }

FunctionQualifiers Function::getFunctionQualifiers() const {
  return qualifiers;
};

void Function::setSignature(FunctionSignature _nature) { signature = _nature; }

void Function::setVisibility(Visibility _vis) { signature.setVisibility(_vis); }

void Function::setBody(std::shared_ptr<BlockExpression> _body) { body = _body; }

size_t Function::getTokens() {
  size_t count = 0;

  llvm::errs() << "Function::getTokens()"
               << "\n";

  llvm::errs() << "Function::getTokens(): " << signature.getTokens() << "\n";

  if (body)
    llvm::errs() << "Function::getTokens(): " << body->getTokens() << "\n";

  count += signature.getTokens();

  if (body)
    count += body->getTokens();

  return count; // fn
};

} // namespace rust_compiler::ast
