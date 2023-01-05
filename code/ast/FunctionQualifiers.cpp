#include "AST/FunctionQualifiers.h"

#include <cassert>

namespace rust_compiler::ast {

void FunctionQualifiers::setAsync() { isAsync = true; }

void FunctionQualifiers::setConst() { isConst = true; }

void FunctionQualifiers::setUnsafe() { isUnsafe = true; }

void FunctionQualifiers::setExtern(std::string_view _abi) {
  isExtern = true;
  abi = _abi;
}

size_t FunctionQualifiers::getTokens() {
  assert(false);
  return 0;
}

} // namespace rust_compiler::ast
