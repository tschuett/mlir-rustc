#include "AST/Function.h"

#include "AST/BlockExpression.h"
#include "AST/Visiblity.h"
#include "llvm/Support/raw_ostream.h"

namespace rust_compiler::ast {

bool Function::hasBody() const {
  return (bool)body;
}

std::shared_ptr<BlockExpression> Function::getBody() { return body; }

void Function::setBody(std::shared_ptr<BlockExpression> _body) { body = _body; }


} // namespace rust_compiler::ast
