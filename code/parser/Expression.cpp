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
#include "AST/MacroInvocationExpression.h"
#include "AST/MatchExpression.h"
#include "AST/MethodCallExpression.h"
#include "AST/NegationExpression.h"
#include "AST/OuterAttribute.h"
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
#include "Location.h"
#include "Parser/Parser.h"
#include "Parser/Precedence.h"
#include "Parser/Restrictions.h"
#include "llvm/Support/YAMLTraits.h"

#include <cassert>
#include <memory>
#include <optional>
#include <string>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace llvm;

namespace rust_compiler::parser {

bool Parser::checkExpressionWithBlock() {
  if (checkOuterAttribute()) {
    llvm::Expected<std::vector<ast::OuterAttribute>> outerAttributes =
        parseOuterAttributes();
    if (auto e = outerAttributes.takeError()) {
      llvm::errs() << "failed to parse outer attributes: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
  }

  if (checkKeyWord(KeyWordKind::KW_IF) &&
      checkKeyWord(KeyWordKind::KW_LET, 1)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_IF)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_MATCH)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_UNSAFE) &&
             check(TokenKind::BraceOpen, 1)) {
  } else if (check(TokenKind::BraceOpen)) {
    return true;
  }

  return false;
}

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseLiteralExpression(std::span<ast::OuterAttribute> outer) {
  Location loc = getLocation();
  LiteralExpression lit = {loc};

  if (check(TokenKind::CHAR_LITERAL))
    lit.setKind(LiteralExpressionKind::CharLiteral);
  else if (check(TokenKind::STRING_LITERAL))
    lit.setKind(LiteralExpressionKind::StringLiteral);
  else if (check(TokenKind::RAW_STRING_LITERAL))
    lit.setKind(LiteralExpressionKind::RawStringLiteral);
  else if (check(TokenKind::BYTE_LITERAL))
    lit.setKind(LiteralExpressionKind::ByteLiteral);
  else if (check(TokenKind::BYTE_STRING_LITERAL))
    lit.setKind(LiteralExpressionKind::ByteStringLiteral);
  else if (check(TokenKind::RAW_BYTE_STRING_LITERAL))
    lit.setKind(LiteralExpressionKind::RawByteStringLiteral);
  else if (check(TokenKind::INTEGER_LITERAL))
    lit.setKind(LiteralExpressionKind::IntegerLiteral);
  else if (check(TokenKind::FLOAT_LITERAL))
    lit.setKind(LiteralExpressionKind::FloatLiteral);
  else if (checkKeyWord(KeyWordKind::KW_TRUE))
    lit.setKind(LiteralExpressionKind::True);
  else if (checkKeyWord(KeyWordKind::KW_FALSE))
    lit.setKind(LiteralExpressionKind::False);
  else {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse literal  in literal expression");
  }
  lit.setStorage(getToken().getLiteral());

  assert(eat(getToken().getKind()));

  return std::make_shared<LiteralExpression>(lit);
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseMacroInvocationExpression() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  MacroInvocationExpression macro = {loc};

  llvm::Expected<ast::SimplePath> path = parseSimplePath();
  if (auto e = path.takeError()) {
    llvm::errs()
        << "failed to parse simple path in macro invocation expression : "
        << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  macro.setPath(*path);

  if (!check(TokenKind::Not))
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse ! token in macro invocation expression");
  assert(eat(TokenKind::Not));

  llvm::Expected<std::shared_ptr<ast::DelimTokenTree>> token =
      parseDelimTokenTree();
  if (auto e = token.takeError()) {
    llvm::errs()
        << "failed to parse delimt token tree in macro invocation expression : "
        << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  macro.setTree(*token);

  return std::make_shared<MacroInvocationExpression>(macro);
}

bool Parser::checkRangeTerminator() {
  if (check(TokenKind::Colon))
    return true;
  if (check(TokenKind::Semi))
    return true;
  if (check(TokenKind::ParenClose))
    return true;
  if (check(TokenKind::SquareClose))
    return true;
  if (check(TokenKind::BraceClose))
    return true;
  return false;
  // heuristic :, or ; or ) or ] or )
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseRangeExpression() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  RangeExpression range = {loc};

  if (check(TokenKind::DotDot)) {
    assert(eat(TokenKind::DotDot));
    if (checkRangeTerminator()) {
      // done
      range.setKind(RangeExpressionKind::RangeFullExpr);
      return std::make_shared<RangeExpression>(range);
    } else {
      // parse
      Restrictions restrictions;
      llvm::Expected<std::shared_ptr<ast::Expression>> right =
          parseExpression(Precedence::DotDot, {}, restrictions);
      if (auto e = right.takeError()) {
        llvm::errs() << "failed to parse expression in range expression: "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      range.setRight(*right);
      range.setKind(RangeExpressionKind::RangeToExpr);
      return std::make_shared<RangeExpression>(range);
    }
  } else if (check(TokenKind::DotDotEq)) {
    assert(eat(TokenKind::DotDotEq));
    // parse
    Restrictions restrictions;
    llvm::Expected<std::shared_ptr<ast::Expression>> right =
        parseExpression(Precedence::DotDotEq, {}, restrictions);
    if (auto e = right.takeError()) {
      llvm::errs() << "failed to parse expression in range expression: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    range.setRight(*right);
    range.setKind(RangeExpressionKind::RangeToInclusiveExpr);
    return std::make_shared<RangeExpression>(range);
  }

  Restrictions restrictions;
  llvm::Expected<std::shared_ptr<ast::Expression>> left =
      parseExpression(Precedence::, {}, restrictions);
  if (auto e = left.takeError()) {
    llvm::errs() << "failed to parse expression in range expression: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  return parseRangeExpression(*left);
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseRangeExpression(std::shared_ptr<ast::Expression> l) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  RangeExpression range = {loc};

  range.setLeft(l);

  if (check(TokenKind::DotDot)) {
    assert(check(TokenKind::DotDot));
    if (checkRangeTerminator()) {
      range.setKind(RangeExpressionKind::RangeFromExpr);
      return std::make_shared<RangeExpression>(range);
    } else {
      llvm::Expected<std::shared_ptr<ast::Expression>> left = parseExpression();
      if (auto e = left.takeError()) {
        llvm::errs() << "failed to parse expression in range expression: "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      range.setRight(*left);
    }
    range.setKind(RangeExpressionKind::RangeExpr);
    return std::make_shared<RangeExpression>(range);
  } else if (check(TokenKind::DotDotEq)) {
    assert(check(TokenKind::DotDotEq));
    range.setKind(RangeExpressionKind::RangeInclusiveExpr);
    llvm::Expected<std::shared_ptr<ast::Expression>> left = parseExpression();
    if (auto e = left.takeError()) {
      llvm::errs() << "failed to parse expression in range expression: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    range.setRight(*left);
    return std::make_shared<RangeExpression>(range);
  }
  return createStringError(inconvertibleErrorCode(),
                           "failed to range expression");
}

Result<ArrayElements, std::string>
Parser::parseArrayElements(std::span<OuterAttribute>,
                           Restrictions restrictions) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  ArrayElements el = {loc};

  Result<std::shared_ptr<ast::Expression>, std::string> first =
      parseExpression({}, restrictions);
  if (auto e = first.takeError()) {
    llvm::errs() << "failed to parse expression in array elements: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  if (check(TokenKind::Semi)) {
    assert(eat(TokenKind::Semi));
    Result<std::shared_ptr<ast::Expression>, std::string> second =
        parseExpression({}, restrictions);
    if (!second) {
      llvm::errs() << "failed to parse expression in array elements: "
                   << second.getError() << "\n";
      printFunctionStack();
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
        return Result<ArrayElements, std::string>(
            "failed to parse array elements: eof");
      } else if (check(TokenKind::Comma) && check(TokenKind::SquareClose, 1)) {
        assert(eat(TokenKind::Comma));
        return el;
      } else if (check(TokenKind::SquareClose)) {
        return el;
      } else if (check(TokenKind::Comma)) {
        assert(eat(TokenKind::Comma));
        Result<std::shared_ptr<ast::Expression>, std::string> next =
            parseExpression({}, restrictions);
        if (!next) {
          llvm::errs() << "failed to parse expression in array elements: "
                       << next.getError() << "\n";
          printFunctionStack();
          exit(EXIT_FAILURE);
        }
        el.addElement(next.getValue());
      }
    }
  }
  return Result<ArrayElements, std::string>("failed to parse array elements");
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseArrayExpression() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
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
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  UnderScoreExpression under = {loc};

  if (!check(TokenKind::Underscore))
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse _ token in underscore expression");
  assert(eat(TokenKind::Underscore));

  return std::make_shared<UnderScoreExpression>(under);
}

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseCallExpression(std::shared_ptr<ast::Expression> e,
                            std::span<OuterAttribute>,
                            Restrictions restrictions) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  CallExpression call = {loc};

  call.setFunction(e);

  if (!check(TokenKind::ParenOpen)) {
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse ( token in call expression");
  }
  assert(eat(TokenKind::ParenOpen));

  if (check(TokenKind::ParenClose)) {
    assert(eat(TokenKind::ParenClose));
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        std::make_shared<CallExpression>(call));
  }

  Result<ast::CallParams, std::string> params = parseCallParams(restrictions);
  if (!params) {
    llvm::errs() << "failed to parse call params in call expression: "
                 << params.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  call.setParams(params.getValue());

  if (!check(TokenKind::ParenClose)) {
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse ) token in call expression");
  }
  assert(eat(TokenKind::ParenClose));

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<CallExpression>(call));
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseTupleIndexingExpression(std::shared_ptr<ast::Expression> lhs) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
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
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
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

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseIndexExpression(std::shared_ptr<ast::Expression> left,
                             std::span<OuterAttribute>,
                             Restrictions restrictions) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  IndexExpression idx = {loc};

  idx.setLeft(left);

  if (!check(TokenKind::SquareOpen))
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse [ token in index expression");
  assert(eat(TokenKind::SquareOpen));

  Result<std::shared_ptr<ast::Expression>, std::string> expr =
      parseExpression({}, restrictions);
  if (!expr) {
    llvm::errs() << "failed to parse expression in index expression: "
                 << expr.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  idx.setRight(expr.getValue());

  if (!check(TokenKind::SquareClose))
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse ] token in index expression");
  assert(eat(TokenKind::SquareClose));

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<IndexExpression>(idx));
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseTypeCastExpression(std::shared_ptr<ast::Expression> lhs) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
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

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseAssignmentExpression(std::shared_ptr<ast::Expression> lhs,
                                  std::span<OuterAttribute> outer,
                                  Restrictions restrictions) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  AssignmentExpression ass = {loc};

  ass.setLeft(lhs);

  if (!check(TokenKind::Eq))
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse = token in assignment expression");
  assert(eat(TokenKind::Eq));

  Result<std::shared_ptr<ast::Expression>, std::string> rhs =
      parseExpression(Precedence::Assign, {}, restrictions);
  if (!rhs) {
    llvm::errs() << "failed to parse expression in assignment expression: "
                 << rhs.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  ass.setRight(rhs.getValue());

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<AssignmentExpression>(ass));
}

Result<ast::CallParams, std::string>
Parser::parseCallParams(Restrictions restrictions) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  CallParams param = {loc};

  Result<std::shared_ptr<ast::Expression>, std::string> expr =
      parseExpression({}, restrictions);
  if (!expr) {
    llvm::errs() << "failed to parse expression in call params: "
                 << expr.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  param.addParam(expr.getValue());

  while (true) {
    if (check(TokenKind::Eof)) {
      return Result<ast::CallParams, std::string>(
          "failed to parse call params: eof");
    } else if (check(TokenKind::ParenClose)) {
      return Result<ast::CallParams, std::string>(param);
    } else if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
      assert(eat(TokenKind::Comma));
      return Result<ast::CallParams, std::string>(param);
    } else if (check(TokenKind::Comma) && !check(TokenKind::ParenClose, 1)) {
      assert(eat(TokenKind::Comma));
      Result<std::shared_ptr<ast::Expression>, std::string> expr =
          parseExpression({}, restrictions);
      if (!expr) {
        llvm::errs() << "failed to parse expression in call params: "
                     << expr.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      param.addParam(expr.getValue());
    } else {
      return Result<ast::CallParams, std::string>(
          "failed to parse call params");
    }
  }
  return Result<ast::CallParams, std::string>("failed to parse call params");
}

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseMethodCallExpression(std::shared_ptr<ast::Expression> receiver,
                                  std::span<OuterAttribute>,
                                  Restrictions restrictions) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  MethodCallExpression call = {loc};

  call.setReceiver(receiver);

  if (!check(TokenKind::Dot)) {
    return Result<std::shared_ptr<ast::Expression>, std::string>(
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
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse ( token in method call expression");
  }
  assert(eat(TokenKind::ParenOpen));

  if (check(TokenKind::ParenClose)) {
    assert(eat(TokenKind::ParenClose));
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        std::make_shared<MethodCallExpression>(call));
  } else if (check(TokenKind::ParenClose)) {
    Result<ast::CallParams, std::string> params = parseCallParams(restrictions);
    if (!params) {
      llvm::errs() << "failed to parse call param in method call expression: "
                   << params.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    call.setCallParams(params.getValue());
    if (!check(TokenKind::ParenClose))
      return Result<std::shared_ptr<ast::Expression>, std::string>(
          "failed to parse ) token in method call expression");
    assert(eat(TokenKind::ParenClose));

    return Result<std::shared_ptr<ast::Expression>, std::string>(
        std::make_shared<MethodCallExpression>(call));
  }
  return Result<std::shared_ptr<ast::Expression>, std::string>(
      "failed to parse method call expression");
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseLazyBooleanExpression(std::shared_ptr<ast::Expression> e,
                                   Restrictions restrictions) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
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

  Result<std::shared_ptr<ast::Expression>> first =
      parseExpression({}, restrictions);
  if (!first) {
    llvm::errs() << "failed to parse expression in lazy boolean expression: "
                 << first.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  laz.setRhs(first.getValue());

  return std::make_shared<LazyBooleanExpression>(laz);
}

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseComparisonExpression(std::shared_ptr<ast::Expression> lhs,
                                  Restrictions restrictions) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  ComparisonExpression comp = {loc};

  comp.setLhs(lhs);

  Precedence pred;
  if (check(TokenKind::EqEq)) {
    comp.setKind(ComparisonExpressionKind::Equal);
    pred = Precedence::Equal;
  } else if (check(TokenKind::Ne)) {
    pred = Precedence::NotEqual;
    comp.setKind(ComparisonExpressionKind::NotEqual);
  } else if (check(TokenKind::Gt)) {
    pred = Precedence::GreaterThan;
    comp.setKind(ComparisonExpressionKind::GreaterThan);
  } else if (check(TokenKind::Lt)) {
    pred = Precedence::LessThan;
    comp.setKind(ComparisonExpressionKind::LessThan);
  } else if (check(TokenKind::Ge)) {
    pred = Precedence::GreaterThanOrEqualTo;
    comp.setKind(ComparisonExpressionKind::GreaterThanOrEqualTo);
  } else if (check(TokenKind::Le)) {
    pred = Precedence::LessThanOrEqualTo;
    comp.setKind(ComparisonExpressionKind::LessThanOrEqualTo);
  } else {
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse kind in comparison expression");
  }

  Result<std::shared_ptr<ast::Expression>, std::string> first =
      parseExpression(pred, {}, restrictions);
  if (!first) {
    llvm::errs() << "failed to parse expression in comparison expression: "
                 << first.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  comp.setRhs(first.getValue());

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<ComparisonExpression>(comp));
}

Result<TupleElements, std::string>
Parser::parseTupleElements(Restrictions restrictions) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  TupleElements tuple = {loc};

  Result<std::shared_ptr<ast::Expression>, std::string> first =
      parseExpression({}, restrictions);
  if (!first) {
    llvm::errs() << "failed to parse expression in tuple elements: "
                 << first.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }

  tuple.addElement(first.getValue());

  if (!check(TokenKind::Comma))
    return Result<TupleElements, std::string>(
        "failed to parse comma token in tuple elements");
  assert(eat(TokenKind::Comma));

  while (true) {
    if (check(TokenKind::Eof)) {
      return Result<TupleElements, std::string>(
          "failed to parse tuple elements: eof");
    } else if (check(TokenKind::ParenClose)) {
      return Result<TupleElements, std::string>(tuple);
    } else {
      Result<std::shared_ptr<ast::Expression>, std::string> next =
          parseExpression({}, restrictions);
      if (!next) {
        llvm::errs() << "failed to parse expression in tuple elements: "
                     << next.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      tuple.addElement(next.getValue());
      if (check(TokenKind::ParenClose))
        continue;
      if (!check(TokenKind::Comma))
        return Result<TupleElements, std::string>(
            "failed to parse comma token in tuple elements");
      assert(eat(TokenKind::Comma));
    }
  }
  return Result<TupleElements, std::string>("failed to parse tuple elements");
}

// FIXME Comma

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseTupleExpression(Restrictions restrictions) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  TupleExpression tuple = {loc};

  if (!check(TokenKind::ParenOpen))
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse ( token in tuple expression");
  assert(eat(TokenKind::ParenOpen));

  Result<TupleElements, std::string> tupleEl = parseTupleElements(restrictions);
  if (!tupleEl) {
    llvm::errs() << "failed to parse tuple elements in tuple expression: "
                 << tupleEl.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  tuple.setElements(tupleEl.getValue());

  if (!check(TokenKind::ParenClose)) {
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse ) token in tuple expression");
    assert(eat(TokenKind::ParenOpen));
  }

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<TupleExpression>(tuple));
}

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseGroupedExpression(Restrictions restrictions) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  GroupedExpression group = {loc};

  if (!check(TokenKind::ParenOpen))
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse ( token in grouped expression");
  assert(eat(TokenKind::ParenOpen));

  Result<std::shared_ptr<ast::Expression>, std::string> first =
      parseExpression({}, restrictions);
  if (!first) {
    llvm::errs() << "failed to parse tuple elements in grouped expression: "
                 << first.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  group.setExpression(first.getValue());

  if (!check(TokenKind::ParenClose)) {
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse ) token in grouped expression");
    assert(eat(TokenKind::ParenOpen));
  }

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<GroupedExpression>(group));
}

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseGroupedOrTupleExpression(Restrictions restrictions) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  CheckPoint cp = getCheckPoint();

  if (!check(TokenKind::ParenOpen))
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse ( token in grouped or tuple expression");
  assert(eat(TokenKind::ParenOpen));

  Result<std::shared_ptr<ast::Expression>, std::string> first =
      parseExpression({}, restrictions);
  if (!first) {
    llvm::errs()
        << "failed to parse expression in grouped or tuple expression: "
        << first.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }

  if (check(TokenKind::ParenClose)) {
    recover(cp);
    return parseGroupedExpression(restrictions);
  }

  return parseTupleExpression(restrictions);
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

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseCompoundAssignmentExpression(std::shared_ptr<ast::Expression> e,
                                          Restrictions restrictions) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  CompoundAssignmentExpression comp = {loc};

  Precedence pred;
  if (check(TokenKind::PlusEq)) {
    comp.setKind(CompoundAssignmentExpressionKind::Add);
    pred = Precedence::PlusAssign;
  } else if (check(TokenKind::MinusEq)) {
    comp.setKind(CompoundAssignmentExpressionKind::Sub);
    pred = Precedence::MinusAssign;
  } else if (check(TokenKind::StarEq)) {
    comp.setKind(CompoundAssignmentExpressionKind::Mul);
    pred = Precedence::MulAssign;
  } else if (check(TokenKind::SlashEq)) {
    comp.setKind(CompoundAssignmentExpressionKind::Div);
    pred = Precedence::DivAssign;
  } else if (check(TokenKind::PercentEq)) {
    comp.setKind(CompoundAssignmentExpressionKind::Rem);
    pred = Precedence::RemAssign;
  } else if (check(TokenKind::CaretEq)) {
    comp.setKind(CompoundAssignmentExpressionKind::Xor);
    pred = Precedence::XorAssign;
  } else if (check(TokenKind::AndEq)) {
    comp.setKind(CompoundAssignmentExpressionKind::And);
    pred = Precedence::AndAssign;
  } else if (check(TokenKind::OrEq)) {
    comp.setKind(CompoundAssignmentExpressionKind::Or);
    pred = Precedence::OrAssign;
  } else if (check(TokenKind::ShlEq)) {
    comp.setKind(CompoundAssignmentExpressionKind::Shl);
    pred = Precedence::ShlAssign;
  } else if (check(TokenKind::ShrEq)) {
    comp.setKind(CompoundAssignmentExpressionKind::Shr);
    pred = Precedence::ShrAssign;
  } else {
    return Result("failed to parse token in compound assignment expression");
    //    return createStringError(
    //        inconvertibleErrorCode(),
    //        "failed to parse token in compound assignment expression");
  }

  Result<std::shared_ptr<ast::Expression>, std::string> expr =
      parseExpression(pred, {}, restrictions);
  if (!expr) {
    llvm::errs()
        << "failed to parse expression in compound assignment expression: "
        << expr.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  comp.setRhs(expr.getValue());

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<CompoundAssignmentExpression>(comp));
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseErrorPropagationExpression(std::shared_ptr<ast::Expression> er) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
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
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
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

Result<ast::Scrutinee, std::string> Parser::parseScrutinee() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  Scrutinee scrut = {loc};

  Restrictions restrictions;
  restrictions.canBeStructExpr = false;

  Result<std::shared_ptr<ast::Expression>, std::string> expr =
      parseExpression({}, restrictions);
  if (!expr) {
    llvm::errs() << "failed to parse expression in scrutinee: "
                 << expr.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }

  scrut.setExpression(expr.getValue());

  return Result<ast::Scrutinee, std::string>(scrut);
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseUnsafeBlockExpression() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
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
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
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

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseBreakExpression(std::span<OuterAttribute>) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  BreakExpression breakExpr = {loc};

  if (checkKeyWord(KeyWordKind::KW_BREAK)) {
    assert(eatKeyWord(KeyWordKind::KW_BREAK));
  } else {
    // check error
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse break keyword");
  }

  if (check(TokenKind::LIFETIME_OR_LABEL)) {
    assert(eat(TokenKind::LIFETIME_OR_LABEL));
    // do something
    breakExpr.setLifetime(getToken());
  }

  if (check(TokenKind::Semi)) {
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        std::make_shared<BreakExpression>(breakExpr));
  } else {
    Restrictions restrictions;
    Result<std::shared_ptr<ast::Expression>, std::string> expr =
        parseExpression({}, restrictions);
    if (!expr) {
      llvm::errs() << "failed to parse expression in return expression: "
                   << expr.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    breakExpr.setExpression(expr.getValue());
  }

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<BreakExpression>(breakExpr));
}

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseIfLetExpression(std::span<OuterAttribute> outer) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  IfLetExpression ifLet = {loc};

  if (checkKeyWord(KeyWordKind::KW_IF)) {
    assert(eatKeyWord(KeyWordKind::KW_IF));
  } else {
    return Result<std::shared_ptr<ast::Expression>, std::string>(

        "failed to parse if key word in if let expression");
  }

  if (checkKeyWord(KeyWordKind::KW_LET)) {
    assert(eatKeyWord(KeyWordKind::KW_LET));
  } else {
    return Result<std::shared_ptr<ast::Expression>, std::string>(
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
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse = token in if let expression");
  }

  Result<ast::Scrutinee, std::string> scrutinee = parseScrutinee();
  if (!scrutinee) {
    llvm::errs() << "failed to parse scrutinee in if let expression: "
                 << scrutinee.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  ifLet.setScrutinee(scrutinee.getValue());

  Result<std::shared_ptr<ast::Expression>, std::string> block =
    parseBlockExpression({});
  if (!block) {
    llvm::errs() << "failed to parse block expression in if let expression: "
                 << block.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  ifLet.setBlock(block.getValue());

  if (checkKeyWord(KeyWordKind::KW_ELSE)) {
    assert(eatKeyWord(KeyWordKind::KW_ELSE));
  } else {
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        std::make_shared<IfLetExpression>(ifLet));
    // done
  }

  // FIXME
  if (checkKeyWord(KeyWordKind::KW_IF) &&
      checkKeyWord(KeyWordKind::KW_LET, 1)) {
    Result<std::shared_ptr<ast::Expression>, std::string> ifLetExpr =
        parseIfLetExpression({});
    if (!ifLetExpr) {
      llvm::errs() << "failed to parse if let expression in if let expression: "
                   << ifLetExpr.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    ifLet.setIfLet(ifLetExpr.getValue());
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        std::make_shared<IfLetExpression>(ifLet));
  } else if (checkKeyWord(KeyWordKind::KW_IF) &&
             !checkKeyWord(KeyWordKind::KW_LET, 1)) {
    Result<std::shared_ptr<ast::Expression>, std::string> ifExpr =
        parseIfExpression({});
    if (!ifExpr) {
      llvm::errs() << "failed to parse if expression in if let expression: "
                   << ifExpr.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    ifLet.setIf(ifExpr.getValue());
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        std::make_shared<IfLetExpression>(ifLet));
  }
  Result<std::shared_ptr<ast::Expression>, std::string> block2 =
      parseBlockExpression();
  if (!block2) {
    llvm::errs() << "failed to parse block expression in if let expression: "
                 << block2.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  ifLet.setTailBlock(block2.getValue());
  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<IfLetExpression>(ifLet));
}

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseDereferenceExpression(std::span<OuterAttribute>) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  DereferenceExpression defer = {loc};

  if (check(TokenKind::Star)) {
    assert(eat(TokenKind::Star));
  } else {
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse * token in dereference expression");
  }

  Restrictions enteredFromUnary;
  enteredFromUnary.enteredFromUnary = true;
  enteredFromUnary.canBeStructExpr = false;
  Result<std::shared_ptr<ast::Expression>, std::string> expr =
      parseExpression(Precedence::UnaryAsterisk, {}, enteredFromUnary);
  if (!expr) {
    llvm::errs() << "failed to parse expression in dereference expression: "
                 << expr.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }

  defer.setExpression(expr.getValue());

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<DereferenceExpression>(defer));
}

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseIfExpression() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  IfExpression ifExpr = {loc};

  if (checkKeyWord(KeyWordKind::KW_IF)) {
    assert(eatKeyWord(KeyWordKind::KW_IF));
  } else {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse if expression");
  }

  Result<std::shared_ptr<ast::Expression>, std::string> cond =
      parseExpression({}, restrictions);
  if (!cond) {
    llvm::errs() << "failed to parse condition in if expression: "
                 << cond.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  ifExpr.setCondition(cond.getValue());

  Result<std::shared_ptr<ast::Expression>, std::string> block =
      parseBlockExpression();
  if (!block) {
    llvm::errs() << "failed to parse block in if expression: "
                 << block.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  ifExpr.setBlock(block.getValue());

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
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  BorrowExpression borrow = {loc};

  bool isMutable = false;

  if (check(TokenKind::And)) {
    assert(eat(TokenKind::And));
  } else if (check(TokenKind::AndAnd)) {
    assert(eat(TokenKind::AndAnd));
  }

  if (checkKeyWord(KeyWordKind::KW_MUT)) {
    isMutable = true;
    assert(eatKeyWord(KeyWordKind::KW_MUT));
    borrow.setMut();
  }

  Restrictions enteredFromUnary;
  enteredFromUnary.enteredFromunary = true;
  enteredFromUnary.canBeStructExpr = false;

  Precedence pred = Precedence::UnaryAnd;
  if (isMutable)
    pred = Precedence::UnaryAndMut;

  llvm::Expected<std::shared_ptr<ast::Expression>> expr =
      parseExpression(pred, {}, enteredFromUnary);
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
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
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
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
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
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();

  ReturnExpression ret = {loc};

  llvm::outs() << "parseReturnExpression"
               << "\n";

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

  llvm::outs() << "end of return expression: "
               << Token2String(getToken().getKind()) << "\n";

  return std::make_shared<ReturnExpression>(ret);
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseExpressionWithBlock() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
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
    return parseInfiniteLoopExpression(std::nullopt);

  if (checkKeyWord(KeyWordKind::KW_WHILE) &&
      checkKeyWord(KeyWordKind::KW_LET, 1))
    return parsePredicatePatternLoopExpression(std::nullopt);

  if (checkKeyWord(KeyWordKind::KW_WHILE) &&
      !checkKeyWord(KeyWordKind::KW_LET, 1))
    return parsePredicatePatternLoopExpression(std::nullopt);

  if (checkKeyWord(KeyWordKind::KW_FOR))
    return parseIteratorLoopExpression(std::nullopt);

  /// FIXME
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse expression with block");
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseExpression(Restrictions restrictions) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  return parseExpression(Precedence::Lowest, restrictions);
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseExpression(Precedence rightBindingPower,
                        Restrictions restrictions) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  CheckPoint cp = getCheckPoint();

  llvm::outs() << "parseExpression"
               << "\n";

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
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();

  BlockExpression bloc = {loc};

  llvm::outs() << "parseBlockExpression"
               << "\n";

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
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};

  llvm::outs() << "parseExpressionWithoutBlock"
               << "\n";

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
    return parseLiteralExpression();
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

  return parsePathInExpressionOrStructExprStructOrStructExprTupleOrStructExprUnitOrMacroInvocationOrExpressionWithPostfix();

  //  /*
  //    PathInExpression -> PathExpression, StructExprStruct, StructExprTuple,
  //    StructExprUnit SimplePath -> MacroInvocation
  //    ExpressionWithPostFix
  //   */
  //
  //  if (check(TokenKind::PathSep) || check(TokenKind::Identifier) ||
  //      checkKeyWord(KeyWordKind::KW_SUPER) ||
  //      checkKeyWord(KeyWordKind::KW_SELFVALUE) ||
  //      checkKeyWord(KeyWordKind::KW_CRATE) ||
  //      checkKeyWord(KeyWordKind::KW_SELFTYPE) ||
  //      checkKeyWord(KeyWordKind::KW_DOLLARCRATE)) {
  //    //    return
  //    //
  //    parsePathInExpressionOrStructExprStructOrStructTupleUnitOrMacroInvocationExpressionOrExpressionWithPostfix();
  //  }
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseExpressionWithPostfix(
    ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
    llvm::Expected<std::shared_ptr<ast::Expression>> left,
    Restrictions restrictions) {

  llvm::outs() << "parseExpressionWithPostfix"
               << "\n";

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
    Restrictions restrictions;
    xxx;
    return parseArithmeticOrLogicalExpression(*left, restrictions);
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
    return parseCompoundAssignmentExpression(*left, restrictions);
  } else if (check(TokenKind::DotDot) || check(TokenKind::DotDotEq)) {
    return parseRangeExpression(*left);
  }

  llvm::outs() << "parseExpressiionWithPostfix: "
               << Token2String(getToken().getKind()) << "\n";
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse "
                           "ExpressionWithPostFix: eof");
}

llvm::Expected<std::shared_ptr<ast::Expression>> Parser::
    parsePathInExpressionOrStructExprStructOrStructExprTupleOrStructExprUnitOrMacroInvocationOrExpressionWithPostfix() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  CheckPoint cp = getCheckPoint();

  while (true) {
    llvm::outs() << "parseXXX1: " << Token2String(getToken().getKind()) << "\n";
    if (check(TokenKind::Eof)) {
      // abort
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse "
                               "PathInExpressionOrStructExprStructOrStructTuple"
                               "UnitOrMacroInvocationExpression: eof");
    } else if (checkSimplePathSegment()) {
      assert(eatSimplePathSegment());
    } else if (check(TokenKind::Lt)) {
      recover(cp);
      return parsePathInExpressionOrStructOrExpressionWithPostfix();
    } else if (checkPathIdentSegment()) {
      recover(cp);
      return parsePathInExpressionOrStructOrExpressionWithPostfix();
    } else if (check(TokenKind::Not)) {
      recover(cp);
      return parseMacroInvocationExpression();
    } else if (check(TokenKind::BraceOpen)) {
      recover(cp);
      return parseStructExpression();
    } else if (check(TokenKind::ParenOpen)) {
      recover(cp);
      return parseStructExpression();
    } else if (checkPostFix()) {
      recover(cp);
      xxx;
      endless loop return parseExpressionWithPostfix();
    } else if (check(TokenKind::PathSep)) {
      assert(eat(TokenKind::PathSep));
    }
  }
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parsePathInExpressionOrStructOrExpressionWithPostfix() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  CheckPoint cp = getCheckPoint();

  while (true) {
    llvm::outs() << "parseXXX2: " << Token2String(getToken().getKind()) << "\n";
    if (check(TokenKind::Eof)) {
      // abort
      return createStringError(
          inconvertibleErrorCode(),
          "failed to parse PathInExpressionOrStructOrExpressionWithPostfix: "
          "eof");
    } else if (check(TokenKind::BraceOpen)) {
      recover(cp);
      return parseStructExpression();
    } else if (check(TokenKind::ParenOpen)) {
      recover(cp);
      return parseStructExpression();
    } else if (check(TokenKind::PathSep)) {
      assert(eat(TokenKind::PathSep));
      //    } else if (check(TokenKind::Lt)) {
      //      xxx;
    } else if (checkPostFix()) {
      recover(cp);
      return parseExpressionWithPostfix();
    }
  }
}

} // namespace rust_compiler::parser
