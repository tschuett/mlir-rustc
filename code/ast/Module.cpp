#include "AST/Module.h"

namespace rust_compiler::ast {

size_t Module::getTokens() {
  return 3; //  +  FIXME
}

std::span<std::shared_ptr<Function>> Module::getFuncs() {
  return std::span<std::shared_ptr<Function>>(funs);
}

} // namespace rust_compiler::ast
