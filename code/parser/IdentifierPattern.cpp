#include "IdentifierPattern.h"

#include "Lexer/KeyWords.h"

#include <optional>

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<patterns::IdentifierPattern>
tryParseIdentifierPattern(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  patterns::IdentifierPattern pattern;

  if (view.front().isKeyWord()) {
    if (view.front().getKeyWordKind() == KeyWordKind::KW_REF) {
      pattern.setRef();
    }
    if (view.front().getKeyWordKind() == KeyWordKind::KW_MUT) {
      pattern.setMut();
    }
  }

  if (view[1].isKeyWord()) {
    if (view[1].getKeyWordKind() == KeyWordKind::KW_REF) {
      pattern.setRef();
    }
    if (view[1].getKeyWordKind() == KeyWordKind::KW_MUT) {
      pattern.setMut();
    }
  }

  if (view.front().isIdentifier()) {
    pattern.setIdentifier(view.front().getIdentifier());
    return pattern;
  }

  if (view[1].isIdentifier()) {
    pattern.setIdentifier(view[1].getIdentifier());
    return pattern;
  }

  if (view[2].isIdentifier()) {
    pattern.setIdentifier(view[2].getIdentifier());
    return pattern;
  }

  // FIXME add PatternNoTopAlt
  return std::nullopt;
}

} // namespace rust_compiler::parser
