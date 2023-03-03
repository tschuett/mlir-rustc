#include "Parser/Parser.h"
#include "Parser/ErrorStack.h"

#include <span>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace llvm;

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseBinaryExpression(std::shared_ptr<ast::Expression> left,
                              std::span<ast::OuterAttribute> outer,
                              Restrictions restrictions) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};

  if (check(TokenKind::QMark)) {
    return parseErrorPropagationExpression(left, outer);
  } else if (check(TokenKind::Plus)) {
    return parseArithmeticOrLogicalExpression(left, outer, restrictions);
  } else if (check(TokenKind::Minus)) {
    return parseArithmeticOrLogicalExpression(left, outer, restrictions);
  }
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseUnaryExpression(std::span<ast::OuterAttribute> outer,
                             Restrictions restrictions) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};

  if (check(TokenKind::Star)) {
    return parseDereferenceExpression();
  } else if (check(TokenKind::And) && check(TokenKind::AndAnd)) {
    return parseBorrowExpression();
  }
}

} // namespace rust_compiler::parser
