#include "AST/FunctionSignature.h"

namespace rust_compiler::ast {

std::shared_ptr<Type> Argument::getType() { return type; }

std::string FunctionSignature::getName() { return name; }

std::shared_ptr<Type> FunctionSignature::getResult() { return resultType; }

Location FunctionSignature::getLocation() { return location; }

} // namespace rust_compiler::ast
