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

  void FunctionSignature::setWhereClause(std::shared_ptr<WhereClause> _where) { where = _where; }

void FunctionSignature::setGenericParams(std::shared_ptr<GenericParams> _gen) {
  gen = _gen;
}

size_t FunctionSignature::getTokens() {
  size_t count = 0;
  count += qual.getTokens();
  ++count; // fn
  ++count; // name
  if (gen)
    count += gen->getTokens();
  count += parm.getTokens() + 2; // ( )
  if (returnType)
    count += returnType->getTokens() + 1; // ->
  if (where)
    count += where->getTokens();

  return count;
}

} // namespace rust_compiler::ast
