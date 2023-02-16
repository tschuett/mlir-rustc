#include "AST/AsyncBlockExpression.h"
#include "AST/BorrowExpression.h"
#include "AST/BreakExpression.h"
#include "AST/ContinueExpression.h"
#include "AST/DereferenceExpression.h"
#include "AST/IfExpression.h"
#include "AST/IfLetExpression.h"
#include "AST/MatchExpression.h"
#include "AST/NegationExpression.h"
#include "AST/ReturnExpression.h"
#include "AST/UnsafeBlockExpression.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <cassert>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace llvm;

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseUnsafeBlockExpression() {
  Location loc = getLocation();
  UnsafeBlockExpression unsafeExpr = {loc};

  if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
    assert(eatKeyWord(KeyWordKind::KW_UNSAFE));
  } else {
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse unsafe token in unsafe block expression");
  }

  llvm::Expected<std::shared_ptr<ast::Expression>> block =
      parseBlockExpression();
  if (auto e = block.takeError()) {
    llvm::errs() << "failed to parse block in unsafe block expression: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  unsafeExpr.setBlock(*block);

  return std::make_shared<UnsafeBlockExpression>(unsafeExpr);
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseAsyncBlockExpression() {
  Location loc = getLocation();
  AsyncBlockExpression asyncExpr = {loc};

  if (checkKeyWord(KeyWordKind::KW_ASYNC)) {
    assert(eatKeyWord(KeyWordKind::KW_ASYNC));
  } else {
    return createStringError(inconvertibleErrorCode(),
                             "failed to async token in async expression");
  }

  if (checkKeyWord(KeyWordKind::KW_MOVE)) {
    assert(eatKeyWord(KeyWordKind::KW_MOVE));
    asyncExpr.setMove();
  }

  llvm::Expected<std::shared_ptr<ast::Expression>> block =
      parseBlockExpression();
  if (auto e = block.takeError()) {
    llvm::errs() << "failed to parse block in async block expression: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  asyncExpr.setBlock(*block);

  return std::make_shared<AsyncBlockExpression>(asyncExpr);
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseBreakExpression() {
  Location loc = getLocation();
  BreakExpression breakExpr = {loc};

  if (checkKeyWord(KeyWordKind::KW_BREAK)) {
    assert(eatKeyWord(KeyWordKind::KW_BREAK));
  } else {
    // check error
  }

  if (check(TokenKind::LIFETIME_OR_LABEL)) {
    assert(eat(TokenKind::LIFETIME_OR_LABEL));
    // do something
  }

  if (check(TokenKind::Semi)) {
    return std::make_shared<BreakExpression>(breakExpr);
  } else {
    llvm::Expected<std::shared_ptr<ast::Expression>> expr = parseExpression();
    if (auto e = expr.takeError()) {
      llvm::errs() << "failed to parse expression in return expression: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    breakExpr.setExpression(*expr);
  }

  return std::make_shared<BreakExpression>(breakExpr);
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseIfLetExpression() {
  Location loc = getLocation();
  IfLetExpression ifLet = {loc};

  if (checkKeyWord(KeyWordKind::KW_IF)) {
    assert(eatKeyWord(KeyWordKind::KW_IF));
  } else {
    // check error
  }

  if (checkKeyWord(KeyWordKind::KW_LET)) {
    assert(eatKeyWord(KeyWordKind::KW_LET));
  } else {
    // check error
  }

  llvm::Expected<ast::patterns::Pattern> pattern = parsePattern();
  // check error

  if (check(TokenKind::Eq)) {
    assert(eat(TokenKind::Eq));
  } else {
    // check error
  }

  llvm::Expected<ast::Scrutinee> scrutinee = parseScrutinee();
  if (auto e = scrutinee.takeError()) {
    llvm::errs() << "failed to parse scrutinee in if let expression: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  llvm::Expected<std::shared_ptr<ast::Expression>> block =
      parseBlockExpression();
  // check error

  if (checkKeyWord(KeyWordKind::KW_ELSE)) {
    assert(eatKeyWord(KeyWordKind::KW_ELSE));
  } else {
    // done
  }

  // FIXME
  if (checkKeyWord(KeyWordKind::KW_IF) &&
      checkKeyWord(KeyWordKind::KW_LET, 1)) {
    llvm::Expected<std::shared_ptr<ast::Expression>> ifLetExpr =
        parseIfLetExpression();
    // check error
  } else if (checkKeyWord(KeyWordKind::KW_IF) &&
             !checkKeyWord(KeyWordKind::KW_LET, 1)) {
    llvm::Expected<std::shared_ptr<ast::Expression>> ifExpr =
        parseIfExpression();
    // check error
  } else {
    llvm::Expected<std::shared_ptr<ast::Expression>> block =
        parseBlockExpression();
    // check error
  }
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseDereferenceExpression() {
  Location loc = getLocation();
  DereferenceExpression defer = {loc};

  if (check(TokenKind::Star)) {
    assert(eat(TokenKind::Star));
  } else {
    // check error
  }

  llvm::Expected<std::shared_ptr<ast::Expression>> expr = parseExpression();
  if (auto e = expr.takeError()) {
    llvm::errs() << "failed to parse expression in dereference expression: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  defer.setExpression(*expr);

  return std::make_shared<DereferenceExpression>(defer);
}

llvm::Expected<std::shared_ptr<ast::Expression>> Parser::parseIfExpression() {
  Location loc = getLocation();
  IfExpression ifExpr = {loc};

  if (checkKeyWord(KeyWordKind::KW_IF)) {
    assert(eatKeyWord(KeyWordKind::KW_IF));
  } else {
    // check error
  }

  llvm::Expected<std::shared_ptr<ast::Expression>> cond = parseExpression();
  if (auto e = cond.takeError()) {
    llvm::errs() << "failed to parse condition in if expression: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  ifExpr.setCondition(*cond);

  llvm::Expected<std::shared_ptr<ast::Expression>> block =
      parseBlockExpression();
  if (auto e = block.takeError()) {
    llvm::errs() << "failed to parse block in if expression: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  ifExpr.setBlock(*block);

  if (checkKeyWord(KeyWordKind::KW_ELSE)) {
    assert(eatKeyWord(KeyWordKind::KW_ELSE));
  } else {
    // without else
    return std::make_shared<IfExpression>(ifExpr);
  }

  // FIXME
  if (checkKeyWord(KeyWordKind::KW_IF) &&
      checkKeyWord(KeyWordKind::KW_LET, 1)) {
    llvm::Expected<std::shared_ptr<ast::Expression>> ifLetExpr =
        parseIfLetExpression();
    // check error
  } else if (checkKeyWord(KeyWordKind::KW_IF) &&
             !checkKeyWord(KeyWordKind::KW_LET, 1)) {
    llvm::Expected<std::shared_ptr<ast::Expression>> ifExpr =
        parseIfExpression();
    // check error
  } else {
    llvm::Expected<std::shared_ptr<ast::Expression>> block =
        parseBlockExpression();
    // check error
  }
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseBorrowExpression() {
  Location loc = getLocation();
  BorrowExpression borrow = {loc};

  if (check(TokenKind::And)) {
    assert(eat(TokenKind::And));
  } else if (check(TokenKind::AndAnd)) {
    assert(eat(TokenKind::AndAnd));
  }

  if (checkKeyWord(KeyWordKind::KW_MUT)) {
    assert(eatKeyWord(KeyWordKind::KW_MUT));
    borrow.setMut();
  }

  llvm::Expected<std::shared_ptr<ast::Expression>> expr = parseExpression();
  if (auto e = expr.takeError()) {
    llvm::errs() << "failed to parse borrow tail expression: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  borrow.setExpression(*expr);

  return std::make_shared<BorrowExpression>(borrow);
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseNegationExpression() {
  Location loc = getLocation();
  NegationExpression neg = {loc};

  if (check(TokenKind::Minus)) {
    neg.setMinus();
    assert(eat());
  } else if (check(TokenKind::Not)) {
    neg.setNot();
    assert(eat());
  } else {
    return createStringError(inconvertibleErrorCode(),
                             "failed to negation token in negation expression");
  }

  llvm::Expected<std::shared_ptr<ast::Expression>> expr = parseExpression();
  if (auto e = expr.takeError()) {
    llvm::errs() << "failed to parse negation tail expression: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  neg.setRight(*expr);

  return std::make_shared<NegationExpression>(neg);
}

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
