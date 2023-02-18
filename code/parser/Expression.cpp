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

llvm::Expected<ast::Scrutinee> Parser::parseScrutinee() {
  Location loc = getLocation();
  Scrutinee scrut = {loc};

  llvm::Expected<std::shared_ptr<ast::Expression>> expr = parseExpression();
  if (auto e = expr.takeError()) {
    llvm::errs() << "failed to parse expression in scrutinee: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  scrut.setExpression(*expr);

  return scrut;
}

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
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse if key word in if let expression");
  }

  if (checkKeyWord(KeyWordKind::KW_LET)) {
    assert(eatKeyWord(KeyWordKind::KW_LET));
  } else {
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse let key word in if let expression");
  }

  llvm::Expected<std::shared_ptr<ast::patterns::Pattern>> pattern =
      parsePattern();
  if (auto e = pattern.takeError()) {
    llvm::errs() << "failed to parse pattern in if let expression: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  ifLet.setPattern(*pattern);

  if (check(TokenKind::Eq)) {
    assert(eat(TokenKind::Eq));
  } else {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse = token in if let expression");
  }

  llvm::Expected<ast::Scrutinee> scrutinee = parseScrutinee();
  if (auto e = scrutinee.takeError()) {
    llvm::errs() << "failed to parse scrutinee in if let expression: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  ifLet.setScrutinee(*scrutinee);

  llvm::Expected<std::shared_ptr<ast::Expression>> block =
      parseBlockExpression();
  if (auto e = block.takeError()) {
    llvm::errs() << "failed to parse block expression in if let expression: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  ifLet.setBlock(*block);

  if (checkKeyWord(KeyWordKind::KW_ELSE)) {
    assert(eatKeyWord(KeyWordKind::KW_ELSE));
  } else {
    return std::make_shared<IfLetExpression>(ifLet);
    // done
  }

  // FIXME
  if (checkKeyWord(KeyWordKind::KW_IF) &&
      checkKeyWord(KeyWordKind::KW_LET, 1)) {
    llvm::Expected<std::shared_ptr<ast::Expression>> ifLetExpr =
        parseIfLetExpression();
    if (auto e = ifLetExpr.takeError()) {
      llvm::errs() << "failed to parse if let expression in if let expression: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    ifLet.setIfLet(*ifLetExpr);
    return std::make_shared<IfLetExpression>(ifLet);
  } else if (checkKeyWord(KeyWordKind::KW_IF) &&
             !checkKeyWord(KeyWordKind::KW_LET, 1)) {
    llvm::Expected<std::shared_ptr<ast::Expression>> ifExpr =
        parseIfExpression();
    if (auto e = ifExpr.takeError()) {
      llvm::errs() << "failed to parse if expression in if let expression: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    ifLet.setIf(*ifExpr);
    return std::make_shared<IfLetExpression>(ifLet);
  }
  llvm::Expected<std::shared_ptr<ast::Expression>> block2 =
      parseBlockExpression();
  if (auto e = block2.takeError()) {
    llvm::errs() << "failed to parse block expression in if let expression: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  ifLet.setTailBlock(*block2);
  return std::make_shared<IfLetExpression>(ifLet);
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
    assert(eat(TokenKind::Minus));
  } else if (check(TokenKind::Not)) {
    neg.setNot();
    assert(eat(TokenKind::Not));
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

  if (check(TokenKind::LIFETIME_OR_LABEL))
    return parseLoopExpression();

  if (checkKeyWord(KeyWordKind::KW_LOOP))
    return parseInfiniteLoopExpression();

  if (checkKeyWord(KeyWordKind::KW_WHILE) &&
      checkKeyWord(KeyWordKind::KW_LET, 1))
    return parsePredicatePatternLoopExpression();

  if (checkKeyWord(KeyWordKind::KW_WHILE) &&
      !checkKeyWord(KeyWordKind::KW_LET, 1))
    return parsePatternLoopExpression();

  if (checkKeyWord(KeyWordKind::KW_FOR))
    return parseIteratorLoopExpression();

  /// FIXME
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse expression with block");
}

llvm::Expected<std::shared_ptr<ast::Expression>> Parser::parseExpression(){

    xxx

}

llvm::Expected<
    std::shared_ptr<ast::Expression>> Parser::parseBlockExpression() {
  Location loc = getLocation();

  BlockExpression bloc = {loc};

  if (!check(TokenKind::BraceOpen)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse { in block expression");
  }

  assert(eat(TokenKind::BraceOpen));

  if (check(TokenKind::Hash) && check(TokenKind::Not, 1) &&
      check(TokenKind::SquareOpen, 2)) {
    llvm::Expected<std::vector<ast::InnerAttribute>> innerAttributes =
        parseInnerAttributes();
    if (auto e = innerAttributes.takeError()) {
      llvm::errs() << "failed to parse inner attributes in block expression: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
  }

  llvm::Expected<ast::Statements> stmts = parseStatements();
  if (auto e = stmts.takeError()) {
    llvm::errs() << "failed to parse statements in block expression: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  bloc.setStatements(*stmts);

  if (!check(TokenKind::BraceClose)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse } in block expression");
  }

  assert(eat(TokenKind::BraceClose));

  return std::make_shared<BlockExpression>(bloc);
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

  if (checkLiteral()) {
    // literal | true | false | ...
    xxx
  }

  if (check(TokenKind::PathSep)) {
    return parsePathInExpressionOrStructExprStructOrStructTupleUnitOrMacroInvocation();
  }

  if (check(TokenKind::SquareOpen)) {
    return parseArrayExpression();
  }

  if (check(TokenKind::And)) {
    return parseBorrowExpression();
  }

  if (check(TokenKind::Lt)) {
    return parseQualifiedPathInExpression();
  }

  if (check(TokenKind::AndAnd)) {
    return parseBorrowExpression();
  }

  if (check(TokenKind::Star)) {
    return parseDereferenceExpression();
  }

  if (check(TokenKind::Not) || check(TokenKind::Minus)) {
    return parseNegationExpression();
  }

  if (check(TokenKind::ParenOpen)) {
    return parseGroupedOrTupleExpression();
  }

  if (checkKeyWord(KeyWordKind::KW_MOVE)) {
    return parseClosureExpression();
  }

  if (check(TokenKind::OrOr)) {
    return parseClosureExpression();
  }

  if (check(TokenKind::Or)) {
    return parseClosureExpression();
  }

  if (check(TokenKind::DotDot)) {
    return parseRangeExpression();
  }

  if (check(TokenKind::DotDotEq)) {
    return parseRangeExpression();
  }

  if (checkKeyWord(KeyWordKind::KW_ASYNC)) {
    return parseAsyncBlockExpression();
  }

  if (checkKeyWord(KeyWordKind::KW_CONTINUE)) {
    return parseContinueExpression();
  }

  if (checkKeyWord(KeyWordKind::KW_BREAK)) {
    return parseBreakExpression();
  }

  if (checkKeyWord(KeyWordKind::KW_RETURN)) {
    return parseReturnExpression();
  }

  if (check(TokenKind::Underscore)) {
    return parseUnderScoreExpression();
  }

  if (check(TokenKind::Identifier) || checkKeyWord(KeyWordKind::KW_SUPER) ||
      checkKeyWord(KeyWordKind::KW_SELFVALUE) ||
      checkKeyWord(KeyWordKind::KW_CRATE) ||
      checkKeyWord(KeyWordKind::KW_SELFTYPE) ||
      checkKeyWord(KeyWordKind::KW_DOLLARCRATE)) {
    return parsePathInExpressionOrStructExprStructOrStructExprUnitOrMacroInvocation();
  }

  return parseExpressionWithPostfix();
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseExpressionWithPostfix() {
  llvm::Expected<std::shared_ptr<ast::Expression>> left = parseExpression();
  if (auto e = left.takeError()) {
    llvm::errs() << "failed to parse expresion in expression with post fix: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  if (check(TokenKind::Dot) && checkKeyWord(KeyWordKind::KW_AWAIT, 1)) {
    return parseAwaitExpression(*left);
  } else if (check(TokenKind::SquareOpen)) {
    return parseIndexingExpression(*left);
  } else if (check(TokenKind::ParenOpen)) {
    return parseCallExpression(*left);
  } else if (check(TokenKind::QMark)) {
    return parseErrorPropagationExpression(*left);
  } else if (check(TokenKind::Dot) && check(TokenKind::Identifier) &&
             !check(TokenKind::ParenOpen)) {
    return parseFieldExpression(*left);
  } else if (check(TokenKind::Dot) && check(TokenKind::INTEGER_LITERAL, 1)) {
    return parseTupleIndexingExpression(*left);
  } else if (check(TokenKind::Dot)) {
    return parseMethodCallExpression(*left);
  } else if (check(TokenKind::Plus) || check(TokenKind::Minus) ||
             check(TokenKind::Star) || check(TokenKind::Slash) ||
             check(TokenKind::Percent) || check(TokenKind::Or) ||
             check(TokenKind::Shl) || check(TokenKind::Shr)) {
    return parseArithmeticOrLogicalExpression(*left);
  } else if (check(TokenKind::EqEq) || check(TokenKind::Ne) ||
             check(TokenKind::Gt) || check(TokenKind::Ge) ||
             check(TokenKind::Le)) {
    return parseComparisonExpression(*left);
  } else if (check(TokenKind::OrOr) || check(TokenKind::AndAnd)) {
    return parseLazyBooleanExpression(*left);
  } else if (checkKeyWord(KeyWordKind::KW_AS)) {
    return parseTypeCastExpression(*left);
  } else if (check(TokenKind::Eq)) {
    return parseAssignmentExpression(*left);
  } else if (check(TokenKind::PlusEq) || check(TokenKind::MinusEq) ||
             check(TokenKind::StarEq) || check(TokenKind::PercentEq) ||
             check(TokenKind::AndEq) || check(TokenKind::OrEq) ||
             check(TokenKind::CaretEq) || check(TokenKind::ShlEq) ||
             check(TokenKind::ShrEq)) {
    return parseCompoundAssignmentExpression(*left);
  }

  return parseRangeExpression(*left);
}

} // namespace rust_compiler::parser
