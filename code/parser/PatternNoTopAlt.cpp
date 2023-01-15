#include "PatternNoTopAlt.h"

#include "Parser/Parser.h"

#include <optional>

using namespace rust_compiler::ast;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::tryParsePatternNoTopAlt(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  std::optional<std::shared_ptr<ast::patterns::PatternNoTopAlt>> literal =
      tryParseLiteralPattern(view);

  if (literal)
    return *literal;

  std::optional<std::shared_ptr<ast::patterns::PatternNoTopAlt>> identifier =
      tryParseIdentifierPattern(view);
  if (identifier)
    return *identifier;

  std::optional<std::shared_ptr<ast::patterns::PatternNoTopAlt>> tuple =
      tryParseTuplePattern(view);
  if (tuple)
    return *tuple;

  std::optional<std::shared_ptr<ast::patterns::PatternNoTopAlt>> rest =
      tryParseRestPattern(view);
  if (rest)
    return *rest;

  // FIXME

  return std::nullopt;
}

} // namespace rust_compiler::parser
