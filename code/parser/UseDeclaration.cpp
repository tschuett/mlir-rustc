#include "UseDeclaration.h"

#include "AST/UseTree.h"

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<UseTree> tryParseUseTree(std::span<Token> tokens) {
  return std::nullopt; // FIXME
}

std::optional<UseDeclaration> tryParseUseDeclaration(std::span<Token> tokens) {
  if (tokens.front().isUseToken()) {
  }
  return std::nullopt; // FIXME
}

} // namespace rust_compiler::parser
