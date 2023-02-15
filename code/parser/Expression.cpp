#include "AST/ReturnExpression.h"
#include "AST/ContinueExpression.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <cassert>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace llvm;

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseContinueExpression() {
  Location loc = getLocation();

  ContinueExpression cont = {loc};

  if (!checkKeyWord(KeyWordKind::KW_CONTINUE))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse continue token");

  assert(eatKeyWord(KeyWordKind::KW_CONTINUE));

  if (check(TokenKind::LIFETIME_OR_LABEL)) {
  }

  // FIXME
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseReturnExpression() {
  Location loc = getLocation();

  ReturnExpression ret = {loc};

  if (!checkKeyWord(KeyWordKind::KW_RETURN))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse return token");

  assert(eatKeyWord(KeyWordKind::KW_RETURN));

  if (check(TokenKind::Semi))
    return std::make_shared<ReturnExpression>(ret);

  llvm::Expected<std::shared_ptr<ast::Expression>> expr = parseExpression();
  if (auto e = expr.takeError()) {
    llvm::errs() << "failed to parse return tail expression: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  ret.setTail(*expr);

  return std::make_shared<ReturnExpression>(ret);
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseExpressionWithBlock() {
  std::vector<ast::OuterAttribute> attributes;
  if (checkOuterAttribute()) {
    llvm::Expected<std::vector<ast::OuterAttribute>> outerAttributes =
        parseOuterAttributes();
    if (auto e = outerAttributes.takeError()) {
      llvm::errs() << "failed to parse outer attributes: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    attributes = *outerAttributes;
  }

  if (check(TokenKind::BraceOpen)) {
    return parseBlockExpression();
  }

  if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
    return parseUnsafeBlockExpression();
  }

  if (checkKeyWord(KeyWordKind::KW_IF) &&
      checkKeyWord(KeyWordKind::KW_LET, 1)) {
    return parseIfLetExpression();
  }

  if (checkKeyWord(KeyWordKind::KW_IF) &&
      !checkKeyWord(KeyWordKind::KW_LET, 1)) {
    return parseIfExpression();
  }

  if (checkKeyWord(KeyWordKind::KW_MATCH)) {
    return parseMatchExpression();
  }
}

llvm::Expected<std::shared_ptr<ast::Expression>> Parser::parseExpression() {}

llvm::Expected<std::shared_ptr<ast::Expression>>
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

  parseStatements();
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseExpressionWithoutBlock() {
  std::vector<ast::OuterAttribute> attributes;
  if (checkOuterAttribute()) {
    llvm::Expected<std::vector<ast::OuterAttribute>> outerAttributes =
        parseOuterAttributes();
    if (auto e = outerAttributes.takeError()) {
      llvm::errs() << "failed to parse outer attributes: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    attributes = *outerAttributes;
  }

  //  if (check(TokenKind::BraceOpen)) {
  //    return parseBlockExpression();
  //  }

  if (checkLiteral()) {
    // literal | true | false
  }

  if (check(TokenKind::PathSep)) {
    // pathinexpression or StructExprStruct or StructTupleUnit
    // simplepath: MAcroInvocation
  }

  if (check(TokenKind::SquareOpen)) {
    // array expression
  }

  if (check(TokenKind::And)) {
    return parseBorrowExpression();
    // borrow
  }

  if (check(TokenKind::Lt)) {
    // return parseQualifiedPathInExpression();
    // borrow
  }

  if (check(TokenKind::AndAnd)) {
    return parseBorrowExpression();
    // borrow
  }

  if (check(TokenKind::Star)) {
    return parseDereferenceExpression();
    // dereference
  }

  if (check(TokenKind::Not) || check(TokenKind::Minus)) {
    return parseNegationExpression();
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

  if (check(TokenKind::DotDot)) {
    // RangeToExpr or RangeFullExpr?
  }

  if (check(TokenKind::DotDotEq)) {
    // RangeToInclusiveExpr
  }

  if (checkKeyWord(KeyWordKind::KW_ASYNC)) {
    return parseAsyncBlockExpression();
    // async block
  }

  if (checkKeyWord(KeyWordKind::KW_CONTINUE)) {
    return parseContinueExpression();
    // continue
  }

  if (checkKeyWord(KeyWordKind::KW_BREAK)) {
    return parseBreakExpression();
    // break
  }

  if (checkKeyWord(KeyWordKind::KW_RETURN)) {
    return parseReturnExpression();
    // return
  }

  if (check(TokenKind::Underscore)) {
    // underscore expression
  }

  /*
    for many / rest (which)
    parseExpression and check next tokens
   */
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
