#include "Lexer/KeyWords.h"
#include "Parser/Parser.h"

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

llvm::Expected<ast::Statements> Parser::parseStatements() {
  // eof

  // statement or exprwithoutblock ?

  while (!check(TokenKind::CurlyClose)) {
    if (checkKeyWord(KeyWordKind::KW_LET)) {
      parseLetStatement();
    }
  }
}

} // namespace rust_compiler::parser
