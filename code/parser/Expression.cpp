#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <cassert>

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::Expression>> Parser::parseExpression() {}

llvm::Expected<std::shared_ptr<ast::BlockExpression>>
Parser::parseBlockExpression() {

  if (!check(TokenKind::BraceOpen)) {
    // error
  }

  assert(eat(TokenKind::BraceOpen));

  if (check(TokenKind::Hash) && check(TokenKind::Not, 1) &&
      check(TokenKind::SquareOpen, 2)) {
    llvm::Expected<std::vector<ast::InnerAttribute>> innerAttributes =
        parseInnerAttributes();
    // check error
  }
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseExpressionWithoutBlock() {
  if (checkOuterAttribute()) {
    llvm::Expected<ast::OuterAttribute> outerAttribute = parseOuterAttribute();
    // check error
  }

  if (check(TokenKind::BraceOpen)) {
    // block expression
  }

  if (checkLiteral()) {
    // literal | true | false
  }

  if (check(TokenKind::PathSep)) {
    // pathinexpression or StructExprStruct or StructTupleUnit
    // simplepath
  }

  if (check(TokenKind::SquareOpen)) {
    // array expression
  }

  if (check(TokenKind::And)) {
    // borrow
  }

  if (check(TokenKind::AndAnd)) {
    // borrow
  }

  if (check(TokenKind::Star)) {
    // dereference
  }

  if (check(TokenKind::Not) || check(TokenKind::Minus)) {
    // negation
  }

  if (check(TokenKind::ParenOpen)) {
    // tuple or grouped
  }

  if (checkKeyWord(KeyWordKind::KW_MOVE)) {
    return parseClosureExpression();
  }

  if (check(TokenKind::OrOr)) {
    return parseClosureExpression();
  }

  if (check(TokenKind::Or)) {
    // closure ?
  }

  if (checkKeyWord(KeyWordKind::KW_ASYNC)) {
    // async block
  }

  if (checkKeyWord(KeyWordKind::KW_CONTINUE)) {
    // continue
  }

  if (checkKeyWord(KeyWordKind::KW_BREAK)) {
    // break
  }

  if (checkKeyWord(KeyWordKind::KW_RETURN)) {
    // return
  }

  if (check(TokenKind::Underscore)) {
    // underscore expression
  }
}

} // namespace rust_compiler::parser


/*
  AwaitExpression
  IndexExpression :
  TupleIndexingExpression :
  StructExpression :
  CallExpression :
  MethodCallExpression :
  FieldExpression :
  PathExpression
  ErrorPropagation
  ArithmeticOrLogicalExpression
  ComparisonExpression
  LazyBooleanExpression
  TypeCastExpression
  AssignmentExpression
  CompoundAssignmentExpression :
  RangeExpression :
  MacroInvocation :
 */


/* checkIdentifier
   PathInExpression
 */

