#include "PatternNoTopAlt.h"

#include "LiteralPattern.h"

#include <optional>

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
tryParsePatternNoTopAlt(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  std::optional<std::shared_ptr<ast::patterns::PatternNoTopAlt>> literal =
      tryParseLiteralPattern(view);

  if (literal)
    return *literal;

  std::optional<std::shared_ptr<ast::patterns::PatternNoTopAlt>> identifier =
      tryParseIdentifierPattern(view);
  if (identifier)
    return *identifier;

  // FIXME

  return std::nullopt;
}

} // namespace rust_compiler::parser
