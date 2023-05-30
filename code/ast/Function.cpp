#include "AST/Function.h"

namespace rust_compiler::ast {

void Function::setParameters(FunctionParameters _functionParameters) {
  functionParameters = _functionParameters;
}

void Function::setQualifiers(FunctionQualifiers _qualifiers) {
  qualifiers = _qualifiers;
}

void Function::setGenericParams(GenericParams _genericParams) {
  genericParams = _genericParams;
}

void Function::setBody(std::shared_ptr<Expression> _body) { body = _body; }

void Function::setWhereClasue(WhereClause _whereClause) {
  whereClause = _whereClause;
}

} // namespace rust_compiler::ast
