#include "Visibility.h"

#include "AST/Visiblity.h"
#include "SimplePath.h"

namespace rust_compiler::lexer {

std::optional<Visibility> tryParseVisibility(std::span<Token> tokens) {
  if (tokens.front().isPubToken()) {
    if (tokens[1].getKind() != TokenKind::ParenOpen) {
      return Visibility();
    }
  }

  return std::nullopt; // FIXME
}

} // namespace rust_compiler::lexer
