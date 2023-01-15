#include "AST/Patterns/Pattern.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

using namespace rust_compiler::ast::patterns;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::patterns::Pattern>>
Parser::tryParsePattern(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  if (view.front().getKind() == lexer::TokenKind::Or)
    view = view.subspan(1);

  std::optional<std::shared_ptr<ast::patterns::PatternNoTopAlt>> noTop =
      tryParsePatternNoTopAlt(view);
  if (noTop) {
    Pattern pat = {tokens.front().getLocation()};
    pat.addPattern(*noTop);

    return std::make_shared<ast::patterns::Pattern>(pat);
  }

  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
