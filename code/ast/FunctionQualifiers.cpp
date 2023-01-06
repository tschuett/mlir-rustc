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
  size_t count = 0;

  if (isAsync)
    ++count;
  if (isConst)
    ++count;
  if (isUnsafe)
    ++count;
  if (isExtern)
    count += 2;

  return count;
}

} // namespace rust_compiler::ast
