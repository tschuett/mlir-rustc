#include "AST/ArrayElements.h"
#include "AST/ArrayExpression.h"
#include "AST/AssignmentExpression.h"
#include "AST/AsyncBlockExpression.h"
#include "AST/AwaitExpression.h"
#include "AST/BorrowExpression.h"
#include "AST/BreakExpression.h"
#include "AST/CallExpression.h"
#include "AST/CallParams.h"
#include "AST/ComparisonExpression.h"
#include "AST/CompoundAssignmentExpression.h"
#include "AST/ContinueExpression.h"
#include "AST/DereferenceExpression.h"
#include "AST/ErrorPropagationExpression.h"
#include "AST/FieldExpression.h"
#include "AST/GroupedExpression.h"
#include "AST/IfExpression.h"
#include "AST/IfLetExpression.h"
#include "AST/IndexEpression.h"
#include "AST/LazyBooleanExpression.h"
#include "AST/LiteralExpression.h"
#include "AST/MatchExpression.h"
#include "AST/MethodCallExpression.h"
#include "AST/NegationExpression.h"
#include "AST/RangeExpression.h"
#include "AST/ReturnExpression.h"
#include "AST/TupleElements.h"
#include "AST/TupleExpression.h"
#include "AST/TupleIndexingExpression.h"
#include "AST/TypeCastExpression.h"
#include "AST/UnderScoreExpression.h"
#include "AST/UnsafeBlockExpression.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <cassert>
#include <memory>
#include <optional>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace llvm;

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseRangeExpression(std::shared_ptr<ast::Expression>l) {
  Location loc = getLocation();
  RangeExpression range = {loc};

  range.setLeft(l);

  if (check(TokenKind::DotDot)) {
  } else if (check(TokenKind::DotDotEq)) {
    assert(check(TokenKind::DotDotEq));
    range.setKind(RangeExpressionKind::RangeInclusiveExpr);
  } else {
    // error
  }
}

llvm::Expected<ast::ArrayElements> Parser::parseArrayElements() {
  Location loc = getLocation();
  ArrayElements el = {loc};

  llvm::Expected<std::shared_ptr<ast::Expression>> first = parseExpression();
  if (auto e = first.takeError()) {
    llvm::errs() << "failed to parse expression in array elements: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  if (check(TokenKind::Semi)) {
    assert(eat(TokenKind::Semi));
    llvm::Expected<std::shared_ptr<ast::Expression>> second = parseExpression();
    if (auto e = second.takeError()) {
      llvm::errs() << "failed to parse expression in array elements: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    el.setKind(ArrayElementsKind::Repeated);
    el.setValue(*first);
    el.setCount(*second);
    return el;
  } else if (check(TokenKind::Comma) && check(TokenKind::SquareClose, 1)) {
    assert(eat(TokenKind::Comma));
    el.setKind(ArrayElementsKind::List);
    el.addElement(*first);
    return el;
  } else {
    while (true) {
      if (check(TokenKind::Eof)) {
        return createStringError(inconvertibleErrorCode(),
                                 "failed to parse array elements: eof");
      } else if (check(TokenKind::Comma) && check(TokenKind::SquareClose, 1)) {
        assert(eat(TokenKind::Comma));
        return el;
      } else if (check(TokenKind::SquareClose)) {
        return el;
      } else if (check(TokenKind::Comma)) {
        assert(eat(TokenKind::Comma));
        llvm::Expected<std::shared_ptr<ast::Expression>> next =
            parseExpression();
        if (auto e = next.takeError()) {
          llvm::errs() << "failed to parse expression in array elements: "
                       << toString(std::move(e)) << "\n";
          exit(EXIT_FAILURE);
        }
        el.addElement(*next);
      }
    }
  }
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse array elements");
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseArrayExpression() {
  Location loc = getLocation();
  ArrayExpression array = {loc};

  if (!check(TokenKind::SquareOpen))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse [ token in array elements");
  assert(eat(TokenKind::SquareOpen));
  if (check(TokenKind::SquareClose)) {
    assert(eat(TokenKind::SquareClose));
    return std::make_shared<ArrayExpression>(array);
  }

  llvm::Expected<ast::ArrayElements> elements = parseArrayElements();
  if (auto e = elements.takeError()) {
    llvm::errs() << "failed to parse array elements in array elements: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  array.setElements(*elements);

  if (!check(TokenKind::SquareClose))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse ] token in array elements");
  assert(eat(TokenKind::SquareOpen));
  return std::make_shared<ArrayExpression>(array);
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseUnderScoreExpression() {
  Location loc = getLocation();
  UnderScoreExpression under = {loc};

  if (!check(TokenKind::Underscore))
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse _ token in underscore expression");
  assert(eat(TokenKind::Underscore));

  return std::make_shared<UnderScoreExpression>(under);
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseCallExpression(std::shared_ptr<ast::Expression> e) {
  Location loc = getLocation();
  CallExpression call = {loc};

  call.setFunction(e);

  if (!check(TokenKind::ParenOpen)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse ( token in call expression");
  }
  assert(eat(TokenKind::ParenOpen));

  if (check(TokenKind::ParenClose)) {
    assert(eat(TokenKind::ParenClose));
    return std::make_shared<CallExpression>(call);
  }

  llvm::Expected<ast::CallParams> params = parseCallParams();
  if (auto e = params.takeError()) {
    llvm::errs() << "failed to parse call params in call expression: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  call.setParams(*params);

  if (!check(TokenKind::ParenClose)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse ) token in call expression");
  }
  assert(eat(TokenKind::ParenClose));

  return std::make_shared<CallExpression>(call);
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseTupleIndexingExpression(std::shared_ptr<ast::Expression> lhs) {
  Location loc = getLocation();
  TupleIndexingExpression tuple = {loc};

  tuple.setTuple(lhs);

  if (!check(TokenKind::Dot)) {
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse . token in tuple indexing expression");
  }
  assert(eat(TokenKind::Dot));

  if (!check(TokenKind::INTEGER_LITERAL)) {
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse INTEGER_LITERAL token in tuple indexing expression");
  }
  tuple.setIndex(getToken().getLiteral());
  assert(eat(TokenKind::INTEGER_LITERAL));

  return std::make_shared<TupleIndexingExpression>(tuple);
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseFieldExpression(std::shared_ptr<ast::Expression> l) {
  Location loc = getLocation();
  FieldExpression field = {loc};

  field.setLeft(l);

  if (!check(TokenKind::Dot))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse . token in field expression");
  assert(eat(TokenKind::Dot));

  if (!checkIdentifier())
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse identifier token in field expression");

  field.setIdentifier(getToken().getIdentifier());

  assert(eat(TokenKind::Identifier));

  return std::make_shared<FieldExpression>(field);
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseIndexExpression(std::shared_ptr<ast::Expression> left) {
  Location loc = getLocation();
  IndexExpression idx = {loc};

  idx.setLeft(left);

  if (!check(TokenKind::SquareOpen))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse [ token in index expression");
  assert(eat(TokenKind::SquareOpen));

  llvm::Expected<std::shared_ptr<ast::Expression>> expr = parseExpression();
  if (auto e = expr.takeError()) {
    llvm::errs() << "failed to parse expression in index expression: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  idx.setRight(*expr);

  if (!check(TokenKind::SquareClose))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse ] token in index expression");
  assert(eat(TokenKind::SquareClose));

  return std::make_shared<IndexExpression>(idx);
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseTypeCastExpression(std::shared_ptr<ast::Expression> lhs) {
  Location loc = getLocation();
  TypeCastExpression ty = {loc};

  ty.setLeft(lhs);

  if (!checkKeyWord(KeyWordKind::KW_AS))
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse as keyword in type cast expression");
  assert(eatKeyWord(KeyWordKind::KW_AS));

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> noBounds =
      parseTypeNoBounds();
  if (auto e = noBounds.takeError()) {
    llvm::errs() << "failed to parse type no bpunds in assignment expression: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  ty.setType(*noBounds);

  return std::make_shared<TypeCastExpression>(ty);
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseAssignmentExpression(std::shared_ptr<ast::Expression> lhs) {
  Location loc = getLocation();
  AssignmentExpression ass = {loc};

  ass.setLeft(lhs);

  if (!check(TokenKind::Eq))
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse = token in assignment expression");
  assert(eat(TokenKind::Eq));

  llvm::Expected<std::shared_ptr<ast::Expression>> expr = parseExpression();
  if (auto e = expr.takeError()) {
    llvm::errs() << "failed to parse expression in assignment expression: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  ass.setRight(*expr);

  return std::make_shared<AssignmentExpression>(ass);
}

llvm::Expected<ast::CallParams> Parser::parseCallParams() {
  Location loc = getLocation();
  CallParams param = {loc};

  llvm::Expected<std::shared_ptr<ast::Expression>> expr = parseExpression();
  if (auto e = expr.takeError()) {
    llvm::errs() << "failed to parse expression in call params: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  param.addParam(*expr);

  while (true) {
    if (check(TokenKind::Eof)) {
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse call params: eof");
    } else if (check(TokenKind::ParenClose)) {
      return param;
    } else if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
      assert(eat(TokenKind::Comma));
      return param;
    } else if (check(TokenKind::Comma) && !check(TokenKind::ParenClose, 1)) {
      assert(eat(TokenKind::Comma));
      llvm::Expected<std::shared_ptr<ast::Expression>> expr = parseExpression();
      if (auto e = expr.takeError()) {
        llvm::errs() << "failed to parse expression in call params: "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      param.addParam(*expr);
    } else {
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse call params");
    }
  }
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse call params");
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseMethodCallExpression(std::shared_ptr<ast::Expression> receiver) {
  Location loc = getLocation();
  MethodCallExpression call = {loc};

  call.setReceiver(receiver);

  if (!check(TokenKind::Dot)) {
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse . token in method call expression");
  }
  assert(eat(TokenKind::Dot));

  llvm::Expected<ast::PathExprSegment> segment = parsePathExprSegment();
  if (auto e = segment.takeError()) {
    llvm::errs()
        << "failed to parse path expr segment in method call expression: "
        << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  call.setSegment(*segment);

  if (!check(TokenKind::ParenOpen)) {
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse ( token in method call expression");
  }
  assert(eat(TokenKind::ParenOpen));

  if (check(TokenKind::ParenClose)) {
    assert(eat(TokenKind::ParenClose));
    return std::make_shared<MethodCallExpression>(call);
  } else if (check(TokenKind::ParenClose)) {
    llvm::Expected<ast::CallParams> params = parseCallParams();
    if (auto e = params.takeError()) {
      llvm::errs() << "failed to parse call param in method call expression: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    call.setCallParams(*params);
    if (!check(TokenKind::ParenClose))
      return createStringError(
          inconvertibleErrorCode(),
          "failed to parse ) token in method call expression");
    assert(eat(TokenKind::ParenClose));

    return std::make_shared<MethodCallExpression>(call);
  }
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse method call expression");
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseLazyBooleanExpression(std::shared_ptr<ast::Expression> e) {
  Location loc = getLocation();
  LazyBooleanExpression laz = {loc};

  laz.setLhs(e);

  if (check(TokenKind::OrOr)) {
    laz.setKind(LazyBooleanExpressionKind::Or);
  } else if (check(TokenKind::OrOr)) {
    laz.setKind(LazyBooleanExpressionKind::And);
  } else {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse kind in lazy boolean expression");
  }

  llvm::Expected<std::shared_ptr<ast::Expression>> first = parseExpression();
  if (auto e = first.takeError()) {
    llvm::errs() << "failed to parse expression in lazy boolean expression: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  laz.setRhs(*first);

  return std::make_shared<LazyBooleanExpression>(laz);
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseComparisonExpression(std::shared_ptr<ast::Expression> lhs) {
  Location loc = getLocation();
  ComparisonExpression comp = {loc};

  comp.setLhs(lhs);

  if (check(TokenKind::EqEq)) {
    comp.setKind(ComparisonExpressionKind::Equal);
  } else if (check(TokenKind::Ne)) {
    comp.setKind(ComparisonExpressionKind::NotEqual);
  } else if (check(TokenKind::Gt)) {
    comp.setKind(ComparisonExpressionKind::GreaterThan);
  } else if (check(TokenKind::Lt)) {
    comp.setKind(ComparisonExpressionKind::LessThan);
  } else if (check(TokenKind::Ge)) {
    comp.setKind(ComparisonExpressionKind::GreaterThan);
  } else if (check(TokenKind::Le)) {
    comp.setKind(ComparisonExpressionKind::LessThan);
  } else {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse kind in comparison expression");
  }

  llvm::Expected<std::shared_ptr<ast::Expression>> first = parseExpression();
  if (auto e = first.takeError()) {
    llvm::errs() << "failed to parse expression in comparison expression: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  comp.setRhs(*first);

  return std::make_shared<ComparisonExpression>(comp);
}

llvm::Expected<TupleElements> Parser::parseTupleElements() {
  Location loc = getLocation();
  TupleElements tuple = {loc};

  llvm::Expected<std::shared_ptr<ast::Expression>> first = parseExpression();
  if (auto e = first.takeError()) {
    llvm::errs() << "failed to parse expression in tuple elements: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  tuple.addElement(*first);

  if (!check(TokenKind::Comma))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse comma token in tuple elements");
  assert(eat(TokenKind::Comma));

  while (true) {
    if (check(TokenKind::Eof)) {
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse tuple elements: eof");
    } else if (check(TokenKind::ParenClose)) {
      return tuple;
    } else {
      llvm::Expected<std::shared_ptr<ast::Expression>> next = parseExpression();
      if (auto e = next.takeError()) {
        llvm::errs() << "failed to parse expression in tuple elements: "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      tuple.addElement(*next);
      if (check(TokenKind::ParenClose))
        continue;
      if (!check(TokenKind::Comma))
        return createStringError(
            inconvertibleErrorCode(),
            "failed to parse comma token in tuple elements");
      assert(eat(TokenKind::Comma));
    }
  }
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse tuple elements");
}

// FIXME Comma

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseTupleExpression() {
  Location loc = getLocation();
  TupleExpression tuple = {loc};

  if (!check(TokenKind::ParenOpen))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse ( token in tuple expression");
  assert(eat(TokenKind::ParenOpen));

  llvm::Expected<TupleElements> tupleEl = parseTupleElements();
  if (auto e = tupleEl.takeError()) {
    llvm::errs() << "failed to parse tuple elements in tuple expression: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  tuple.setElements(*tupleEl);

  if (!check(TokenKind::ParenClose)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse ) token in tuple expression");
    assert(eat(TokenKind::ParenOpen));
  }

  return std::make_shared<TupleExpression>(tuple);
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseGroupedExpression() {
  Location loc = getLocation();
  GroupedExpression group = {loc};

  if (!check(TokenKind::ParenOpen))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse ( token in grouped expression");
  assert(eat(TokenKind::ParenOpen));

  llvm::Expected<std::shared_ptr<ast::Expression>> first = parseExpression();
  if (auto e = first.takeError()) {
    llvm::errs() << "failed to parse expression in grouped expression: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  group.setExpression(*first);

  if (!check(TokenKind::ParenClose)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse ) token in grouped expression");
    assert(eat(TokenKind::ParenOpen));
  }

  return std::make_shared<GroupedExpression>(group);
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseGroupedOrTupleExpression() {
  Location loc = getLocation();
  CheckPoint cp = getCheckPoint();

  if (!check(TokenKind::ParenOpen))
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse ( token in grouped or tuple expression");
  assert(eat(TokenKind::ParenOpen));

  llvm::Expected<std::shared_ptr<ast::Expression>> first = parseExpression();
  if (auto e = first.takeError()) {
    llvm::errs()
        << "failed to parse expression in grouped or tuple expression: "
        << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  if (check(TokenKind::ParenClose)) {
    recover(cp);
    return parseGroupedExpression();
  }

  return parseTupleExpression();
}

// llvm::Expected<std::shared_ptr<ast::Expression>>
// Parser::parseTupleIndexingExpression(std::shared_ptr<ast::Expression> e) {
//   Location loc = getLocation();
//   TupleIndexingExpression tuple = {loc};
//
//   if (!check(TokenKind::Dot))
//     return createStringError(
//         inconvertibleErrorCode(),
//         "failed to parse dot token in tuple indexing expression");
//   assert(eat(TokenKind::Dot));
//
//   if (!check(TokenKind::INTEGER_LITERAL))
//     return createStringError(
//         inconvertibleErrorCode(),
//         "failed to parse integer literal token in tuple indexing
//         expression");
//
//   tuple.setIndex(getToken().IntegerLiteral());
//
//   assert(eat(TokenKind::Dot));
// }

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseCompoundAssignmentExpression(std::shared_ptr<ast::Expression> e) {
  Location loc = getLocation();
  CompoundAssignmentExpression comp = {loc};

  if (check(TokenKind::PlusEq))
    comp.setKind(CompoundAssignmentExpressionKind::Add);
  else if (check(TokenKind::MinusEq))
    comp.setKind(CompoundAssignmentExpressionKind::Sub);
  else if (check(TokenKind::StarEq))
    comp.setKind(CompoundAssignmentExpressionKind::Mul);
  else if (check(TokenKind::SlashEq))
    comp.setKind(CompoundAssignmentExpressionKind::Div);
  else if (check(TokenKind::PercentEq))
    comp.setKind(CompoundAssignmentExpressionKind::Rem);
  else if (check(TokenKind::CaretEq))
    comp.setKind(CompoundAssignmentExpressionKind::Xor);
  else if (check(TokenKind::AndEq))
    comp.setKind(CompoundAssignmentExpressionKind::And);
  else if (check(TokenKind::OrEq))
    comp.setKind(CompoundAssignmentExpressionKind::Or);
  else if (check(TokenKind::ShlEq))
    comp.setKind(CompoundAssignmentExpressionKind::Shl);
  else if (check(TokenKind::ShrEq))
    comp.setKind(CompoundAssignmentExpressionKind::Shr);
  else {
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse token in compound assignment expression");
  }

  llvm::Expected<std::shared_ptr<ast::Expression>> expr = parseExpression();
  if (auto e = expr.takeError()) {
    llvm::errs()
        << "failed to parse expression in compound assignment expression: "
        << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  comp.setRhs(*expr);

  return std::make_shared<CompoundAssignmentExpression>(comp);
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseErrorPropagationExpression(std::shared_ptr<ast::Expression> er) {
  Location loc = getLocation();
  ErrorPropagationExpression ep = {loc};

  if (!check(TokenKind::QMark)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse ? token in await expression");
  }
  assert(eat(TokenKind::QMark));

  ep.setLhs(er);

  return std::make_shared<ErrorPropagationExpression>(ep);
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseAwaitExpression(std::shared_ptr<ast::Expression> e) {
  Location loc = getLocation();
  AwaitExpression a = {loc};

  if (!check(TokenKind::Dot)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse : token in await expression");
  }
  assert(eat(TokenKind::Dot));

  if (!checkKeyWord(KeyWordKind::KW_AWAIT)) {
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse await keyword in await expression");
  }
  assert(eatKeyWord(KeyWordKind::KW_AWAIT));

  a.setLhs(e);

  return std::make_shared<AwaitExpression>(a);
}

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
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse if expression");
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
    if (auto e = ifLetExpr.takeError()) {
      llvm::errs() << "failed to parse if let expression in if expression: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    ifExpr.setTrailing(*ifLetExpr);
    return std::make_shared<IfExpression>(ifExpr);
  } else if (checkKeyWord(KeyWordKind::KW_IF) &&
             !checkKeyWord(KeyWordKind::KW_LET, 1)) {
    llvm::Expected<std::shared_ptr<ast::Expression>> ifExprTail =
        parseIfExpression();
    if (auto e = ifExprTail.takeError()) {
      llvm::errs() << "failed to parse if expression in if expression: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    ifExpr.setTrailing(*ifExprTail);
    return std::make_shared<IfExpression>(ifExpr);
  } else {
    llvm::Expected<std::shared_ptr<ast::Expression>> block =
        parseBlockExpression();
    if (auto e = block.takeError()) {
      llvm::errs() << "failed to parse block expression in if expression: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    ifExpr.setTrailing(*block);
    return std::make_shared<IfExpression>(ifExpr);
  }

  // no tail
  return std::make_shared<IfExpression>(ifExpr);
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
    Token tok = getToken();
    cont.setLifetime(tok);
  }

  return std::make_shared<ContinueExpression>(cont);
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

  // FIXME attributes

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
    return parsePredicatePatternLoopExpression();

  if (checkKeyWord(KeyWordKind::KW_FOR))
    return parseIteratorLoopExpression();

  /// FIXME
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse expression with block");
}

llvm::Expected<std::shared_ptr<ast::Expression>> Parser::parseExpression() {
  CheckPoint cp = getCheckPoint();

  if (checkOuterAttribute()) {
    llvm::Expected<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (auto e = outer.takeError()) {
      llvm::errs() << "failed to parse outer attributes in expression: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
  }

  if (checkKeyWord(KeyWordKind::KW_LOOP)) {
    recover(cp);
    return parseExpressionWithBlock();
  } else if (checkKeyWord(KeyWordKind::KW_MATCH)) {
    recover(cp);
    return parseExpressionWithBlock();
  } else if (checkKeyWord(KeyWordKind::KW_WHILE)) {
    recover(cp);
    return parseExpressionWithBlock();
  } else if (checkKeyWord(KeyWordKind::KW_IF)) {
    recover(cp);
    return parseExpressionWithBlock();
  } else if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
    recover(cp);
    return parseExpressionWithBlock();
  } else if (checkKeyWord(KeyWordKind::KW_FOR)) {
    recover(cp);
    return parseExpressionWithBlock();
  } else if (check(TokenKind::BraceOpen)) {
    recover(cp);
    return parseExpressionWithBlock();
  } else if (check(TokenKind::LIFETIME_OR_LABEL) && check(TokenKind::Colon)) {
    recover(cp);
    return parseExpressionWithBlock();
  }

  recover(cp);
  return parseExpressionWithoutBlock();
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseBlockExpression() {
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

  // FIXME attributes

  if (checkLiteral()) {
    Location loc = getLocation();
    Token tok = getToken();
    std::string id = tok.getIdentifier();
    if (check(TokenKind::CHAR_LITERAL)) {
      assert(eat(TokenKind::CHAR_LITERAL));
      return std::make_shared<LiteralExpression>(
          loc, LiteralExpressionKind::CharLiteral, id);
    } else if (check(TokenKind::STRING_LITERAL)) {
      assert(eat(TokenKind::STRING_LITERAL));
      return std::make_shared<LiteralExpression>(
          loc, LiteralExpressionKind::StringLiteral, id);
    } else if (check(TokenKind::RAW_STRING_LITERAL)) {
      assert(eat(TokenKind::RAW_STRING_LITERAL));
      return std::make_shared<LiteralExpression>(
          loc, LiteralExpressionKind::RawStringLiteral, id);
    } else if (check(TokenKind::BYTE_LITERAL)) {
      assert(eat(TokenKind::BYTE_LITERAL));
      return std::make_shared<LiteralExpression>(
          loc, LiteralExpressionKind::ByteLiteral, id);
    } else if (check(TokenKind::BYTE_STRING_LITERAL)) {
      assert(eat(TokenKind::BYTE_STRING_LITERAL));
      return std::make_shared<LiteralExpression>(
          loc, LiteralExpressionKind::ByteStringLiteral, id);
    } else if (check(TokenKind::RAW_BYTE_STRING_LITERAL)) {
      assert(eat(TokenKind::RAW_BYTE_STRING_LITERAL));
      return std::make_shared<LiteralExpression>(
          loc, LiteralExpressionKind::RawByteStringLiteral, id);
    } else if (check(TokenKind::INTEGER_LITERAL)) {
      assert(eat(TokenKind::INTEGER_LITERAL));
      return std::make_shared<LiteralExpression>(
          loc, LiteralExpressionKind::IntegerLiteral, id);
    } else if (check(TokenKind::FLOAT_LITERAL)) {
      assert(eat(TokenKind::FLOAT_LITERAL));
      return std::make_shared<LiteralExpression>(
          loc, LiteralExpressionKind::FloatLiteral, id);
    } else if (checkKeyWord(KeyWordKind::KW_TRUE)) {
      assert(eatKeyWord(KeyWordKind::KW_TRUE));
      return std::make_shared<LiteralExpression>(
          loc, LiteralExpressionKind::True, id);
    } else if (checkKeyWord(KeyWordKind::KW_FALSE)) {
      assert(eatKeyWord(KeyWordKind::KW_FALSE));
      return std::make_shared<LiteralExpression>(
          loc, LiteralExpressionKind::False, id);
    }
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

  if (checkKeyWord(KeyWordKind::KW_MOVE)) {
    return parseClosureExpression();
  }

  if (check(TokenKind::OrOr)) {
    return parseClosureExpression();
  }

  if (check(TokenKind::Or)) {
    return parseClosureExpression();
  }

  if (check(TokenKind::SquareOpen)) {
    return parseArrayExpression();
  }

  if (check(TokenKind::ParenOpen)) {
    return parseGroupedOrTupleExpression();
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

  if (check(TokenKind::PathSep)) {
    /*
      PathInExpression -> PathExpression, StructExprStruct, StructExprTuple,
      StructExprUnit SimplePath -> MacroInvocation
     */
    return parsePathInExpressionOrStructExprStructOrStructTupleUnitOrMacroInvocationExpression();
  }

  if (check(TokenKind::Identifier) || checkKeyWord(KeyWordKind::KW_SUPER) ||
      checkKeyWord(KeyWordKind::KW_SELFVALUE) ||
      checkKeyWord(KeyWordKind::KW_CRATE) ||
      checkKeyWord(KeyWordKind::KW_SELFTYPE) ||
      checkKeyWord(KeyWordKind::KW_DOLLARCRATE)) {
    return parsePathInExpressionOrStructExprStructOrStructTupleUnitOrMacroInvocationExpression();
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
    return parseIndexExpression(*left);
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

llvm::Expected<std::shared_ptr<ast::Expression>> Parser::
    parsePathInExpressionOrStructExprStructOrStructTupleUnitOrMacroInvocationExpression() {
  Location loc = getLocation();

  while (true) {
    if (check(TokenKind::Eof)) {
      // abort
    } else if (check(TokenKind::Eof)) {
    } else if (check(TokenKind::Not)) {
      return parseMacroInvocationExpression();
    } else if (check(TokenKind::BraceOpen)) {
      return parseStructExpression();
    } else if (check(TokenKind::ParenOpen)) {
      return parseStructExpression();
    }
  }
}

} // namespace rust_compiler::parser
