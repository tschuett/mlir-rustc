#include "Parser/Parser.h"

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::Statement>> Parser::parseLetStatement() {
  if (checkOuterAttribute()) {
    llvm::Expected<std::vector<ast::OuterAttribute>> outer = parseOuterAttributes();
    // check error
  }

  if (!checkKeyWord(KeyWordKind::KW_LET)) {
    // errro
  }

  assert(eatKeyWord(KeyWordKind::KW_LET));

  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>> pattern =
      parsePatternNoTopAlt();
  // check error

  if (check(TokenKind::Colon)) {
    assert(eat(TokenKind::Colon));
    llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> type =
        parseType();
    // check error
  }

  if (check(TokenKind::Eq)) {
    assert(eat(TokenKind::Eq));
    llvm::Expected<std::shared_ptr<ast::Expression>> expr = parseExpression();
    // check error
  }

  if (checkKeyWord(KeyWordKind::KW_ELSE)) {
    assert(eatKeyWord(KeyWordKind::KW_ELSE));
    llvm::Expected<std::shared_ptr<ast::BlockExpression>> block =
        parseBlockExpression();
    // check error
  }

    if (!check(TokenKind::Semi)) {
    // errro
  }

  assert(eat(TokenKind::Semi));

}

} // namespace rust_compiler::parser
