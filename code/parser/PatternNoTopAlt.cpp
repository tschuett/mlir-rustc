#include "Lexer/Token.h"
#include "Parser/Parser.h"

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

/// https://doc.rust-lang.org/reference/patterns.html

llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parseReferencePattern() {
  if (!check(lexer::TokenKind::And) && !check(lexer::TokenKind::AndAnd)) {
    // error
  }

  bool And = false;
  bool AndAnd = false;
  bool mut = false;
  if (check(lexer::TokenKind::And)) {
    assert(eat(lexer::TokenKind::And));
    And = true;
  }

  if (check(lexer::TokenKind::AndAnd)) {
    assert(eat(lexer::TokenKind::AndAnd));
    AndAnd = true;
  }

  if (checkKeyWord(KeyWordKind::KW_MUT)) {
    assert(eatKeyWord(lexer::KeyWordKind::KW_MUT));
    mut = true;
  }

  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>> woRange =
      parsePatternWithoutRange();

  // FIXME
}

llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parsePatternNoTopAlt() {

  if (check(lexer::TokenKind::And) || check(lexer::TokenKind::AndAnd))
    return parseReferencePattern();

  if (check(lexer::TokenKind::ParenOpen))
    return parseTupleOrGroupedPattern();

  if (check(lexer::TokenKind::SquareOpen))
    return parseSlicePattern();

  // if (check(lexer::TokenKind::DotDot))
  //   return parseRestPattern();

  if (check(lexer::TokenKind::Underscore))
    return parseWildCardPattern();
}

} // namespace rust_compiler::parser
