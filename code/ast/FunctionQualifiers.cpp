#include "AST/FunctionQualifiers.h"

#include <cassert>

namespace rust_compiler::ast {

void FunctionQualifiers::setAsync() { isAsync = true; }

void FunctionQualifiers::setConst() { isConst = true; }

void FunctionQualifiers::setUnsafe() { isUnsafe = true; }


} // namespace rust_compiler::ast
