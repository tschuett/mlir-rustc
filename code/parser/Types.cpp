#include "Parser/Parser.h"

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

llvm::Expected<ast::types::TypeExpression> Parser::parseType() {
  if (check(TokenKind::ParenOpen))
    return parseTupleOrParensType();

  //    if (check(TokenKind::Not))
  //      return Never

  if (check(TokenKind::Star))
    return parseTypePointer();

  if (check(TokenKind::SquareOpen))
    return parseArrayOrSliceType();

  
}

} // namespace rust_compiler::parser
