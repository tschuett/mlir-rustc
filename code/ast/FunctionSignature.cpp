#include "AST/FunctionSignature.h"

namespace rust_compiler::ast {

std::string FunctionSignature::getName() { return name; }

void FunctionSignature::setName(std::string_view _name) { name = _name; }

void FunctionSignature::setQualifiers(FunctionQualifiers _qual) {
  qual = _qual;
}

void FunctionSignature::setParameters(FunctionParameters _parm) {
  parm = _parm;
}

void FunctionSignature::setReturnType(
    std::shared_ptr<ast::types::Type> _returnType) {
  returnType = _returnType;
}

void FunctionSignature::setWhereClause(WhereClause _where) { where = _where; }

void FunctionSignature::setGenericParams(std::shared_ptr<GenericParams> _gen) {
  gen = _gen;
}

} // namespace rust_compiler::ast
