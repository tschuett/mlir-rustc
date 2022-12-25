#include "Visibility.h"

namespace rust_compiler::ast {

std::optional<Visibility> tryParseVisibility(std::span<Token> tokens) {
  if (tokens.front().isPubToken()) {
    if (tokens[1].getKind() == TokenKind::ParenOpen) {
      
    }
  }

  return std::nullopt; // FIXME
}

} // namespace rust_compiler::ast
