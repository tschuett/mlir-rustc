#include "AST/Function.h"

#include "AST/BlockExpression.h"
#include "AST/Visiblity.h"
#include "llvm/Support/raw_ostream.h"

namespace rust_compiler::ast {

bool Function::hasBody() const { return (bool)body; }

void Function::setParameters(FunctionParameters _functionParameters) {
  functionParameters = _functionParameters;
}

void Function::setQualifiers(FunctionQualifiers _qualifiers) {
  qualifiers = _qualifiers;
}

void Function::setGenericParams(GenericParams _genericParams) {
  genericParams = _genericParams;
}

std::shared_ptr<Expression> Function::getBody() { return body; }

void Function::setBody(std::shared_ptr<Expression> _body) { body = _body; }

void Function::setWhereClasue(WhereClause _whereClause) {
  whereClause = _whereClause;
}

} // namespace rust_compiler::ast
