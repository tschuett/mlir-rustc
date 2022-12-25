#include "Attributes.h"

#include <optional>

namespace rust_compiler {

std::optional<OuterAttribute> tryParseOuterAttribute(std::span<Token> tokens) {
  std::span<Token> view = tokens;

  if (not(view[0].getKind() == TokenKind::Hash and
          view[1].getKind() == TokenKind::SquareOpen)) {
    return std::nullopt;
  } else {
  }

  return std::nullopt;
}

std::optional<InnerAttribute> tryParseInnerAttribute(std::span<Token> tokens) {
  std::span<Token> view = tokens;

  if (not(view[0].getKind() == TokenKind::Hash and
          view[1].getKind() == TokenKind::Exclaim and
          view[2].getKind() == TokenKind::SquareOpen)) {
    return std::nullopt;
  } else {
  }

  return std::nullopt;
}

} // namespace rust_compiler
