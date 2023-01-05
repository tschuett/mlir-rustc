#include "AST/FunctionSignature.h"

namespace rust_compiler::ast {

std::shared_ptr<Type> Argument::getType() { return type; }

std::string FunctionSignature::getName() { return name; }

std::shared_ptr<Type> FunctionSignature::getResult() { return resultType; }

Location FunctionSignature::getLocation() { return location; }

void FunctionSignature::setName(std::string_view _name) { name = _name; }

void FunctionSignature::setQualifiers(FunctionQualifiers _qual) {
  qual = _qual;
}

} // namespace rust_compiler::ast
