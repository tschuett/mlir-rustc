#include "LiteralPattern.h"

#include "AST/Patterns/LiteralPattern.h"
#include "Lexer/Token.h"

namespace rust_compiler::parser {

  std::optional<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
tryParseLiteralPattern(std::span<lexer::Token> tokens) {
//  std::span<lexer::Token> view = tokens;
//
//  if (view.front().getKind() == lexer::TokenKind::DecIntegerLiteral) {
//    LiteralPattern pat = {tokens.front().getLocation()};
//    //pat.
//  }
//
  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
