#include "UseDeclaration.h"

#include "AST/UseTree.h"

namespace rust_compiler::ast {

using namespace rust_compiler::lexer;

std::optional<UseTree> tryParseUseTree(std::span<Token> tokens) {
  return std::nullopt; // FIXME
}

std::optional<UseDeclaration> tryParseUseDeclaration(std::span<Token> tokens) {
  if (tokens.front().isUseToken()) {
  }
  return std::nullopt; // FIXME
}

} // namespace rust_compiler::ast
