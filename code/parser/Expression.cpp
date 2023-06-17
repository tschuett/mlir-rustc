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
#include "AST/LifetimeOrLabel.h"
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

#include <cassert>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
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
    StringResult<std::vector<ast::OuterAttribute>> outerAttributes =
        parseOuterAttributes();
    if (!outerAttributes) {
      llvm::errs()
          << "failed to parse outer attributes in expression with block: "
          << outerAttributes.getError() << "\n";
      printFunctionStack();
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
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse literal  in literal expression");
  }
  lit.setStorage(getToken().getLiteral());

  assert(eat(getToken().getKind()));

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<LiteralExpression>(lit));
}

StringResult<std::shared_ptr<ast::Expression>>
Parser::parseMacroInvocationExpression() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  MacroInvocationExpression macro = {loc};

  StringResult<ast::SimplePath> path = parseSimplePath();
  if (!path) {
    llvm::errs() << "failed to simple path in macro invocation expression: "
                 << path.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  macro.setSimplePath(path.getValue());

  if (!check(TokenKind::Not))
    return StringResult<std::shared_ptr<ast::Expression>>(
        "failed to parse ! token in macro invocation expression");
  assert(eat(TokenKind::Not));

  StringResult<std::shared_ptr<ast::DelimTokenTree>> token =
      parseDelimTokenTree();
  if (!token) {
    llvm::errs()
        << "failed to delim token tree in macro invocation expression: "
        << token.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  macro.setTree(token.getValue());

  return StringResult<std::shared_ptr<ast::Expression>>(
      std::make_shared<MacroInvocationExpression>(macro));
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

Result<ArrayElements, std::string>
Parser::parseArrayElements(std::span<OuterAttribute>,
                           Restrictions restrictions) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  ArrayElements el = {loc};

  Result<std::shared_ptr<ast::Expression>, std::string> first =
      parseExpression({}, restrictions);
  if (!first) {
    llvm::errs() << "failed to parse expression in array elements: "
                 << first.getError() << "\n";
    printFunctionStack();
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
    el.setValue(first.getValue());
    el.setCount(second.getValue());
    return Result<ArrayElements, std::string>(el);
  } else if (check(TokenKind::Comma) && check(TokenKind::SquareClose, 1)) {
    assert(eat(TokenKind::Comma));
    el.setKind(ArrayElementsKind::List);
    el.addElement(first.getValue());
    return Result<ArrayElements, std::string>(el);
  } else {
    el.setKind(ArrayElementsKind::List);
    while (true) {
      if (check(TokenKind::Eof)) {
        return Result<ArrayElements, std::string>(
            "failed to parse array elements: eof");
      } else if (check(TokenKind::Comma) && check(TokenKind::SquareClose, 1)) {
        assert(eat(TokenKind::Comma));
        return Result<ArrayElements, std::string>(el);
      } else if (check(TokenKind::SquareClose)) {
        return Result<ArrayElements, std::string>(el);
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

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseArrayExpression(std::span<OuterAttribute>) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  ArrayExpression array = {loc};

  if (!check(TokenKind::SquareOpen))
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse [ token in array elements");
  assert(eat(TokenKind::SquareOpen));
  if (check(TokenKind::SquareClose)) {
    assert(eat(TokenKind::SquareClose));
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        std::make_shared<ArrayExpression>(array));
  }

  Restrictions restrictions;
  StringResult<ast::ArrayElements> elements =
      parseArrayElements({}, restrictions);
  if (!elements) {
    llvm::errs() << "failed to parse array elements in array expression: "
                 << elements.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }

  array.setElements(elements.getValue());

  if (!check(TokenKind::SquareClose))
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse ] token in array elements");
  assert(eat(TokenKind::SquareClose));
  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<ArrayExpression>(array));
}

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseUnderScoreExpression(std::span<OuterAttribute>) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  UnderScoreExpression under = {loc};

  if (!check(TokenKind::Underscore))
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse _ token in underscore expression");
  assert(eat(TokenKind::Underscore));

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<UnderScoreExpression>(under));
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

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseTupleIndexingExpression(std::shared_ptr<ast::Expression> lhs,
                                     std::span<OuterAttribute> outer,
                                     Restrictions restrictions) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  TupleIndexingExpression tuple = {loc};

  tuple.setTuple(lhs);

  if (!check(TokenKind::Dot)) {
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse . token in tuple indexing expression");
  }
  assert(eat(TokenKind::Dot));

  if (!check(TokenKind::INTEGER_LITERAL)) {
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse INTEGER_LITERAL token in tuple indexing expression");
  }
  tuple.setIndex(getToken().getLiteral());
  assert(eat(TokenKind::INTEGER_LITERAL));

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<TupleIndexingExpression>(tuple));
}

StringResult<std::shared_ptr<ast::Expression>>
Parser::parseFieldExpression(std::shared_ptr<ast::Expression> l,
                             std::span<OuterAttribute>,
                             Restrictions restrictions) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  FieldExpression field = {loc};

  field.setLeft(l);

  if (!check(TokenKind::Dot))
    return StringResult<std::shared_ptr<ast::Expression>>(
        "failed to parse . token in field expression");
  assert(eat(TokenKind::Dot));

  if (!checkIdentifier())
    return StringResult<std::shared_ptr<ast::Expression>>(
        "failed to parse identifier token in field expression");

  field.setIdentifier(getToken().getIdentifier());

  assert(eat(TokenKind::Identifier));

  return StringResult<std::shared_ptr<ast::Expression>>(
      std::make_shared<FieldExpression>(field));
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
    std::string s =
        llvm::formatv(
            "{0}\n{1}",
            "failed to parse expression in index expression: ", expr.getError())
            .str();
    return Result<std::shared_ptr<ast::Expression>, std::string>(expr);
  }
  idx.setRight(expr.getValue());

  if (!check(TokenKind::SquareClose)) {
    std::string s = llvm::formatv("{0}{1}{2}", "failed to parse ] found ",
                                  Token2String(getToken().getKind()),
                                  " token in index expression")
                        .str();
    return Result<std::shared_ptr<ast::Expression>, std::string>(s);
  }
  assert(eat(TokenKind::SquareClose));

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<IndexExpression>(idx));
}

StringResult<std::shared_ptr<ast::Expression>>
Parser::parseTypeCastExpression(std::shared_ptr<ast::Expression> lhs) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  TypeCastExpression ty = {loc};

  ty.setLeft(lhs);

  if (!checkKeyWord(KeyWordKind::KW_AS))
    return StringResult<std::shared_ptr<ast::Expression>>(
        "failed to parse as keyword in type cast expression");
  assert(eatKeyWord(KeyWordKind::KW_AS));

  StringResult<std::shared_ptr<ast::types::TypeExpression>> noBounds =
      parseTypeNoBounds();
  if (!noBounds) {
    llvm::errs()
        << "failed to parse type no bounds expression in type cast expression: "
        << noBounds.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  ty.setType(noBounds.getValue());

  return StringResult<std::shared_ptr<ast::Expression>>(
      std::make_shared<TypeCastExpression>(ty));
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

  //  llvm::errs() << "parseMethodCallExpression"
  //               << "\n";

  call.setReceiver(receiver);

  if (!check(TokenKind::Dot)) {
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse . token in method call expression");
  }
  assert(eat(TokenKind::Dot));

  StringResult<ast::PathExprSegment> segment = parsePathExprSegment();
  if (!segment) {
    llvm::errs()
        << "failed to parse path expr segment in method call expression: "
        << segment.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  call.setSegment(segment.getValue());

  if (!check(TokenKind::ParenOpen)) {
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse ( token in method call expression");
  }
  assert(eat(TokenKind::ParenOpen));

  if (check(TokenKind::ParenClose)) {
    assert(eat(TokenKind::ParenClose));
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        std::make_shared<MethodCallExpression>(call));
  } else {
    Result<ast::CallParams, std::string> params = parseCallParams(restrictions);
    if (!params) {
      llvm::errs() << getToken().getLocation().toString() << "\n";
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
  llvm::errs() << "failed to parse method call expression"
               << "\n";
  llvm::errs() << Token2String(getToken().getKind()) << "\n";
  if (getToken().getKind() == TokenKind::Identifier)
    llvm::errs() << getToken().getIdentifier().toString() << "\n";
  return Result<std::shared_ptr<ast::Expression>, std::string>(
      "failed to parse method call expression");
}

Result<std::shared_ptr<ast::Expression>, std::string>
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
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse kind in lazy boolean expression");
  }

  Result<std::shared_ptr<ast::Expression>, std::string> first =
      parseExpression({}, restrictions);
  if (!first) {
    llvm::errs() << "failed to parse expression in lazy boolean expression: "
                 << first.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  laz.setRhs(first.getValue());

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<LazyBooleanExpression>(laz));
}

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseComparisonExpression(std::shared_ptr<ast::Expression> lhs,
                                  Restrictions restrictions) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  ComparisonExpression comp = {loc};

  // llvm::errs() << "parse comparison expression"
  //              << "\n";
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

  assert(eat(getToken().getKind())); // cheating

  Result<std::shared_ptr<ast::Expression>, std::string> first =
      parseExpression(pred, {}, restrictions);
  if (!first) {
    llvm::errs() << "failed to parse expression in comparison expression: "
                 << first.getError() << "\n";
    printFunctionStack();
    std::string s =
        llvm::formatv("{0} {1}",
                      "failed to parse expression in comparison expression: ",
                      first.getError())
            .str();
    return Result<std::shared_ptr<ast::Expression>, std::string>(s);

    // exit(EXIT_FAILURE);
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

  if (check(TokenKind::ParenClose)) {
    // done
    assert(eat(TokenKind::ParenClose));
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        std::make_shared<TupleExpression>(tuple));
  }

  Result<TupleElements, std::string> tupleEl = parseTupleElements(restrictions);
  if (!tupleEl) {
    llvm::errs() << "failed to parse tuple elements in tuple expression: "
                 << tupleEl.getError() << "\n";
    printFunctionStack();
    std::string s =
        llvm::formatv("{0} {1}",
                      "failed to parse tuple elements in tuple expression: ",
                      tupleEl.getError())
            .str();
    return Result<std::shared_ptr<ast::Expression>, std::string>(s);
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

  if (!check(TokenKind::ParenClose))
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse ) token in grouped expression");
  assert(eat(TokenKind::ParenClose));

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

  if (check(TokenKind::ParenClose)) {
    assert(eat(TokenKind::ParenClose));
    recover(cp);
    return parseTupleExpression(restrictions);
  }

  Result<std::shared_ptr<ast::Expression>, std::string> first =
      parseExpression({}, restrictions);
  if (!first) {
    llvm::errs()
        << "failed to parse expression in grouped or tuple expression: "
        << first.getError() << "\n";
    printFunctionStack();
    // exit(EXIT_FAILURE);
    std::string s =
        llvm::formatv(
            "{0} {1}",
            "failed to parse expression in grouped or tuple expression: ",
            first.getError())
            .str();
    return Result<std::shared_ptr<ast::Expression>, std::string>(s);
  }

  if (check(TokenKind::ParenClose)) {
    recover(cp);
    return parseGroupedExpression(restrictions);
  }

  recover(cp);
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

  comp.setLhs(e);

  Precedence pred;
  if (check(TokenKind::PlusEq)) {
    comp.setKind(CompoundAssignmentExpressionKind::Add);
    pred = Precedence::PlusAssign;
    assert(eat(TokenKind::PlusEq));
  } else if (check(TokenKind::MinusEq)) {
    comp.setKind(CompoundAssignmentExpressionKind::Sub);
    pred = Precedence::MinusAssign;
    assert(eat(TokenKind::MinusEq));
  } else if (check(TokenKind::StarEq)) {
    comp.setKind(CompoundAssignmentExpressionKind::Mul);
    pred = Precedence::MulAssign;
    assert(eat(TokenKind::StarEq));
  } else if (check(TokenKind::SlashEq)) {
    comp.setKind(CompoundAssignmentExpressionKind::Div);
    pred = Precedence::DivAssign;
    assert(eat(TokenKind::SlashEq));
  } else if (check(TokenKind::PercentEq)) {
    comp.setKind(CompoundAssignmentExpressionKind::Rem);
    pred = Precedence::RemAssign;
    assert(eat(TokenKind::PercentEq));
  } else if (check(TokenKind::CaretEq)) {
    comp.setKind(CompoundAssignmentExpressionKind::Xor);
    pred = Precedence::XorAssign;
    assert(eat(TokenKind::CaretEq));
  } else if (check(TokenKind::AndEq)) {
    comp.setKind(CompoundAssignmentExpressionKind::And);
    pred = Precedence::AndAssign;
    assert(eat(TokenKind::AndEq));
  } else if (check(TokenKind::OrEq)) {
    comp.setKind(CompoundAssignmentExpressionKind::Or);
    pred = Precedence::OrAssign;
    assert(eat(TokenKind::OrEq));
  } else if (check(TokenKind::ShlEq)) {
    comp.setKind(CompoundAssignmentExpressionKind::Shl);
    pred = Precedence::ShlAssign;
    assert(eat(TokenKind::ShlEq));
  } else if (check(TokenKind::ShrEq)) {
    comp.setKind(CompoundAssignmentExpressionKind::Shr);
    pred = Precedence::ShrAssign;
    assert(eat(TokenKind::ShrEq));
  } else {
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse token in compound assignment expression");
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

StringResult<std::shared_ptr<ast::Expression>>
Parser::parseErrorPropagationExpression(std::shared_ptr<ast::Expression> er) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  ErrorPropagationExpression ep = {loc};

  if (!check(TokenKind::QMark)) {
    return StringResult<std::shared_ptr<ast::Expression>>(
        "failed to parse ? token in await expression");
  }
  assert(eat(TokenKind::QMark));

  ep.setLhs(er);

  return StringResult<std::shared_ptr<ast::Expression>>(
      std::make_shared<ErrorPropagationExpression>(ep));
}

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseAwaitExpression(std::shared_ptr<ast::Expression> e) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  AwaitExpression a = {loc};

  if (!check(TokenKind::Dot)) {
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse : token in await expression");
  }
  assert(eat(TokenKind::Dot));

  if (!checkKeyWord(KeyWordKind::KW_AWAIT)) {
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse await keyword in await expression");
  }
  assert(eatKeyWord(KeyWordKind::KW_AWAIT));

  a.setLhs(e);

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<AwaitExpression>(a));
}

adt::Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseAwaitExpression(std::shared_ptr<ast::Expression> e,
                             std::span<ast::OuterAttribute>) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  AwaitExpression a = {loc};

  if (!check(TokenKind::Dot)) {
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse : token in await expression");
  }
  assert(eat(TokenKind::Dot));

  if (!checkKeyWord(KeyWordKind::KW_AWAIT)) {
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse await keyword in await expression");
  }
  assert(eatKeyWord(KeyWordKind::KW_AWAIT));

  a.setLhs(e);

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<AwaitExpression>(a));
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

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseUnsafeBlockExpression(std::span<OuterAttribute>) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  UnsafeBlockExpression unsafeExpr = {loc};

  if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
    assert(eatKeyWord(KeyWordKind::KW_UNSAFE));
  } else {
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse unsafe token in unsafe block expression");
  }

  Result<std::shared_ptr<ast::Expression>, std::string> block =
      parseBlockExpression({});
  if (!block) {
    llvm::errs()
        << "failed to parse block expression in unsafe block expression: "
        << block.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  unsafeExpr.setBlock(block.getValue());

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<UnsafeBlockExpression>(unsafeExpr));
}

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseAsyncBlockExpression(std::span<OuterAttribute>) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  AsyncBlockExpression asyncExpr = {loc};

  if (checkKeyWord(KeyWordKind::KW_ASYNC)) {
    assert(eatKeyWord(KeyWordKind::KW_ASYNC));
  } else {
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to async token in async expression");
  }

  if (checkKeyWord(KeyWordKind::KW_MOVE)) {
    assert(eatKeyWord(KeyWordKind::KW_MOVE));
    asyncExpr.setMove();
  }

  Result<std::shared_ptr<ast::Expression>, std::string> block =
      parseBlockExpression({});
  if (!block) {
    llvm::errs()
        << "failed to parse block expression in async block expression: "
        << block.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  asyncExpr.setBlock(block.getValue());

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<AsyncBlockExpression>(asyncExpr));
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
    LifetimeOrLabel l = {loc};
    l.setLifeTime(getToken().getIdentifier());
    breakExpr.setLifetime(l);
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

  StringResult<std::shared_ptr<ast::patterns::Pattern>> pattern =
      parsePattern();
  if (!pattern) {
    llvm::errs() << "failed to parse pattern in if let expression: "
                 << pattern.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }

  ifLet.setPattern(pattern.getValue());

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
      parseBlockExpression({});
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
Parser::parseIfExpression(std::span<OuterAttribute>) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  IfExpression ifExpr = {loc};

  if (checkKeyWord(KeyWordKind::KW_IF)) {
    assert(eatKeyWord(KeyWordKind::KW_IF));
  } else {
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse if expression");
  }

  Restrictions noStructExpr;
  noStructExpr.canBeStructExpr = false;
  Result<std::shared_ptr<ast::Expression>, std::string> cond =
      parseExpression({}, noStructExpr);
  if (!cond) {
    llvm::errs() << "failed to parse condition in if expression: "
                 << cond.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  ifExpr.setCondition(cond.getValue());

  Result<std::shared_ptr<ast::Expression>, std::string> block =
      parseBlockExpression({});
  if (!block) {
    llvm::errs() << Token2String(getToken().getKind()) << "\n";
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
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        std::make_shared<IfExpression>(ifExpr));
  }

  // FIXME
  if (checkKeyWord(KeyWordKind::KW_IF) &&
      checkKeyWord(KeyWordKind::KW_LET, 1)) {
    Result<std::shared_ptr<ast::Expression>, std::string> ifLetExpr =
        parseIfLetExpression({});
    if (!ifLetExpr) {
      llvm::errs() << "failed to parse if let expression in if expression: "
                   << ifLetExpr.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    ifExpr.setTrailing(ifLetExpr.getValue());
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        std::make_shared<IfExpression>(ifExpr));
  } else if (checkKeyWord(KeyWordKind::KW_IF) &&
             !checkKeyWord(KeyWordKind::KW_LET, 1)) {
    Result<std::shared_ptr<ast::Expression>, std::string> ifExprTail =
        parseIfExpression({});
    if (!ifExprTail) {
      llvm::errs() << "failed to parse if expression in if expression: "
                   << ifExprTail.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    ifExpr.setTrailing(ifExprTail.getValue());
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        std::make_shared<IfExpression>(ifExpr));
  } else {
    Result<std::shared_ptr<ast::Expression>, std::string> block =
        parseBlockExpression({});
    if (!block) {
      llvm::errs() << "failed to parse block expression in if expression: "
                   << block.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    ifExpr.setTrailing(block.getValue());
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        std::make_shared<IfExpression>(ifExpr));
  }

  // no tail
  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<IfExpression>(ifExpr));
}

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseBorrowExpression(std::span<OuterAttribute>) {
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
  enteredFromUnary.enteredFromUnary = true;
  enteredFromUnary.canBeStructExpr = false;

  Precedence pred = Precedence::UnaryAnd;
  if (isMutable)
    pred = Precedence::UnaryAndMut;

  Result<std::shared_ptr<ast::Expression>, std::string> expr =
      parseExpression(pred, {}, enteredFromUnary);
  if (!expr) {
    llvm::errs() << "failed to parse borrow tail expression: "
                 << expr.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }

  borrow.setExpression(expr.getValue());

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<BorrowExpression>(borrow));
}

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseNegationExpression(std::span<OuterAttribute>) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  NegationExpression neg = {loc};

  Precedence pred;
  if (check(TokenKind::Minus)) {
    neg.setMinus();
    assert(eat(TokenKind::Minus));
    pred = Precedence::UnaryMinus;
  } else if (check(TokenKind::Not)) {
    neg.setNot();
    pred = Precedence::UnaryNot;
    assert(eat(TokenKind::Not));
  } else {
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to negation token in negation expression");
  }

  Restrictions enteredFromUnary;
  enteredFromUnary.enteredFromUnary = true;
  Result<std::shared_ptr<ast::Expression>, std::string> expr =
      parseExpression(pred, {}, enteredFromUnary);
  if (!expr) {
    llvm::errs() << "failed to parse negation tail expression: "
                 << expr.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }

  neg.setRight(expr.getValue());

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<NegationExpression>(neg));
}

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseContinueExpression(std::span<OuterAttribute>) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();

  ContinueExpression cont = {loc};

  if (!checkKeyWord(KeyWordKind::KW_CONTINUE))
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse continue token");

  assert(eatKeyWord(KeyWordKind::KW_CONTINUE));

  if (check(TokenKind::LIFETIME_OR_LABEL)) {
    Token tok = getToken();
    LifetimeOrLabel l = {getToken().getLocation()};
    l.setLifeTime(tok.getIdentifier());
    assert(eat(TokenKind::LIFETIME_OR_LABEL));
    cont.setLifetime(l);
  }

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<ContinueExpression>(cont));
}

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseReturnExpression(std::span<OuterAttribute>) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();

  ReturnExpression ret = {loc};

  //  llvm::outs() << "parseReturnExpression"
  //               << "\n";

  if (!checkKeyWord(KeyWordKind::KW_RETURN))
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse return token");
  assert(eatKeyWord(KeyWordKind::KW_RETURN));

  if (check(TokenKind::Semi))
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        std::make_shared<ReturnExpression>(ret));

  if (check(TokenKind::Eof)) // for gtest
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        std::make_shared<ReturnExpression>(ret));

  Restrictions restrictions;
  Result<std::shared_ptr<ast::Expression>, std::string> expr =
      parseExpression({}, restrictions);
  if (!expr) {
    llvm::errs() << "failed to parse return tail expression: "
                 << expr.getError() << "\n";
    printFunctionStack();
    std::string s =
        llvm::formatv("{0} {1}", "failed to parse return tail expression: ",
                      expr.getError())
            .str();
    return Result<std::shared_ptr<ast::Expression>, std::string>(s);
    // exit(EXIT_FAILURE);
  }

  ret.setTail(expr.getValue());

  //  llvm::errs() << "end of return expression: "
  //               << Token2String(getToken().getKind()) << "\n";

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<ReturnExpression>(ret));
}

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseExpressionWithBlock(std::span<OuterAttribute> outer) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  //  std::vector<ast::OuterAttribute> attributes;
  //  if (checkOuterAttribute()) {
  //    llvm::Expected<std::vector<ast::OuterAttribute>> outerAttributes =
  //        parseOuterAttributes();
  //    if (auto e = outerAttributes.takeError()) {
  //      llvm::errs() << "failed to parse outer attributes: "
  //                   << toString(std::move(e)) << "\n";
  //      exit(EXIT_FAILURE);
  //    }
  //    attributes = *outerAttributes;
  //  }
  //
  //  // FIXME attributes

  if (check(TokenKind::BraceOpen)) {
    return parseBlockExpression(outer);
  }

  if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
    return parseUnsafeBlockExpression(outer);
  }

  if (checkKeyWord(KeyWordKind::KW_IF) &&
      checkKeyWord(KeyWordKind::KW_LET, 1)) {
    return parseIfLetExpression(outer);
  }

  if (checkKeyWord(KeyWordKind::KW_IF) &&
      !checkKeyWord(KeyWordKind::KW_LET, 1)) {
    return parseIfExpression(outer);
  }

  if (checkKeyWord(KeyWordKind::KW_MATCH)) {
    return parseMatchExpression(outer);
  }

  if (check(TokenKind::LIFETIME_OR_LABEL))
    return parseLoopExpression(outer);

  if (checkKeyWord(KeyWordKind::KW_LOOP))
    return parseInfiniteLoopExpression(outer);

  if (checkKeyWord(KeyWordKind::KW_WHILE) &&
      checkKeyWord(KeyWordKind::KW_LET, 1))
    return parsePredicatePatternLoopExpression(outer);

  if (checkKeyWord(KeyWordKind::KW_WHILE) &&
      !checkKeyWord(KeyWordKind::KW_LET, 1))
    return parsePredicatePatternLoopExpression(outer);

  if (checkKeyWord(KeyWordKind::KW_FOR))
    return parseIteratorLoopExpression(outer);

  /// FIXME
  return Result<std::shared_ptr<ast::Expression>, std::string>(
      "failed to parse expression with block");
}

// Result<std::shared_ptr<ast::Expression>, std::string>
// Parser::parseExpressionWithoutBlock() {
//   ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
//
//   llvm::outs() << "parseExpressionWithoutBlock"
//                << "\n";
//
//   std::vector<ast::OuterAttribute> outer;
//   if (checkOuterAttribute()) {
//     StringResult<std::vector<ast::OuterAttribute>> outerAttributes =
//         parseOuterAttributes();
//     if (!outerAttributes) {
//       llvm::errs()
//           << "failed to parse outer attributes expression without block: "
//           << outerAttributes.getError() << "\n";
//       printFunctionStack();
//       exit(EXIT_FAILURE);
//     }
//     outer = outerAttributes.getValue();
//   }
//
//   //  // FIXME attributes
//
//   if (checkLiteral()) {
//     return parseLiteralExpression(outer);
//   }
//   if (check(TokenKind::And)) {
//     return parseBorrowExpression(outer);
//   }
//
//   if (check(TokenKind::Lt)) {
//     return parseQualifiedPathInExpression(outer);
//   }
//
//   if (check(TokenKind::AndAnd)) {
//     return parseBorrowExpression(outer);
//   }
//
//   if (check(TokenKind::Star)) {
//     return parseDereferenceExpression(outer);
//   }
//
//   if (check(TokenKind::Not) || check(TokenKind::Minus)) {
//     return parseNegationExpression(outer);
//   }
//
//   if (checkKeyWord(KeyWordKind::KW_MOVE)) {
//     return parseClosureExpression(outer);
//   }
//
//   if (check(TokenKind::OrOr)) {
//     return parseClosureExpression(outer);
//   }
//
//   if (check(TokenKind::Or)) {
//     return parseClosureExpression(outer);
//   }
//
//   if (check(TokenKind::SquareOpen)) {
//     return parseArrayExpression(outer);
//   }
//
//   if (check(TokenKind::ParenOpen)) {
//     return parseGroupedOrTupleExpression(outer);
//   }
//
//   if (check(TokenKind::DotDot)) {
//     return parseRangeExpression(outer);
//   }
//
//   if (check(TokenKind::DotDotEq)) {
//     return parseRangeExpression(outer);
//   }
//
//   if (checkKeyWord(KeyWordKind::KW_ASYNC)) {
//     return parseAsyncBlockExpression(outer);
//   }
//
//   if (checkKeyWord(KeyWordKind::KW_CONTINUE)) {
//     return parseContinueExpression(outer);
//   }
//
//   if (checkKeyWord(KeyWordKind::KW_BREAK)) {
//     return parseBreakExpression(outer);
//   }
//
//   if (checkKeyWord(KeyWordKind::KW_RETURN)) {
//     return parseReturnExpression(outer);
//   }
//
//   if (check(TokenKind::Underscore)) {
//     return parseUnderScoreExpression(outer);
//   }
//
//   xxx;
//
//   return
//   parsePathInExpressionOrStructExprStructOrStructExprTupleOrStructExprUnitOrMacroInvocationOrExpressionWithPostfix(
//       outer);
//
//   //  /*
//   //    PathInExpression -> PathExpression, StructExprStruct,
//   StructExprTuple,
//   //    StructExprUnit SimplePath -> MacroInvocation
//   //    ExpressionWithPostFix
//   //   */
//   //
//   //  if (check(TokenKind::PathSep) || check(TokenKind::Identifier) ||
//   //      checkKeyWord(KeyWordKind::KW_SUPER) ||
//   //      checkKeyWord(KeyWordKind::KW_SELFVALUE) ||
//   //      checkKeyWord(KeyWordKind::KW_CRATE) ||
//   //      checkKeyWord(KeyWordKind::KW_SELFTYPE) ||
//   //      checkKeyWord(KeyWordKind::KW_DOLLARCRATE)) {
//   //    //    return
//   //    //
//   //
//   parsePathInExpressionOrStructExprStructOrStructTupleUnitOrMacroInvocationExpressionOrExpressionWithPostfix();
//   //  }
// }

// StringResult<std::shared_ptr<ast::Expression>>
// Parser::parseExpressionWithPostfix(
//     ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
//     llvm::Expected<std::shared_ptr<ast::Expression>> left,
//     Restrictions restrictions) {
//
//   llvm::outs() << "parseExpressionWithPostfix"
//                << "\n";
//
//   if (check(TokenKind::Dot) && checkKeyWord(KeyWordKind::KW_AWAIT, 1)) {
//     return parseAwaitExpression(*left);
//   } else if (check(TokenKind::SquareOpen)) {
//     return parseIndexExpression(*left);
//   } else if (check(TokenKind::ParenOpen)) {
//     return parseCallExpression(*left);
//   } else if (check(TokenKind::QMark)) {
//     return parseErrorPropagationExpression(*left);
//   } else if (check(TokenKind::Dot) && check(TokenKind::Identifier) &&
//              !check(TokenKind::ParenOpen)) {
//     return parseFieldExpression(*left);
//   } else if (check(TokenKind::Dot) && check(TokenKind::INTEGER_LITERAL, 1)) {
//     return parseTupleIndexingExpression(*left);
//   } else if (check(TokenKind::Dot)) {
//     return parseMethodCallExpression(*left);
//   } else if (check(TokenKind::Plus) || check(TokenKind::Minus) ||
//              check(TokenKind::Star) || check(TokenKind::Slash) ||
//              check(TokenKind::Percent) || check(TokenKind::Or) ||
//              check(TokenKind::Shl) || check(TokenKind::Shr)) {
//     Restrictions restrictions;
//     xxx;
//     return parseArithmeticOrLogicalExpression(*left, restrictions);
//   } else if (check(TokenKind::EqEq) || check(TokenKind::Ne) ||
//              check(TokenKind::Gt) || check(TokenKind::Ge) ||
//              check(TokenKind::Le)) {
//     return parseComparisonExpression(*left);
//   } else if (check(TokenKind::OrOr) || check(TokenKind::AndAnd)) {
//     return parseLazyBooleanExpression(*left);
//   } else if (checkKeyWord(KeyWordKind::KW_AS)) {
//     return parseTypeCastExpression(*left);
//   } else if (check(TokenKind::Eq)) {
//     return parseAssignmentExpression(*left);
//   } else if (check(TokenKind::PlusEq) || check(TokenKind::MinusEq) ||
//              check(TokenKind::StarEq) || check(TokenKind::PercentEq) ||
//              check(TokenKind::AndEq) || check(TokenKind::OrEq) ||
//              check(TokenKind::CaretEq) || check(TokenKind::ShlEq) ||
//              check(TokenKind::ShrEq)) {
//     return parseCompoundAssignmentExpression(*left, restrictions);
//   } else if (check(TokenKind::DotDot) || check(TokenKind::DotDotEq)) {
//     return parseRangeExpression(*left);
//   }
//
//   llvm::outs() << "parseExpressiionWithPostfix: "
//                << Token2String(getToken().getKind()) << "\n";
//   return createStringError(inconvertibleErrorCode(),
//                            "failed to parse "
//                            "ExpressionWithPostFix: eof");
// }

// StringResult<std::shared_ptr<ast::Expression>> Parser::
//     parsePathInExpressionOrStructExprStructOrStructExprTupleOrStructExprUnitOrMacroInvocationOrExpressionWithPostfix()
//     {
//   ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
//   CheckPoint cp = getCheckPoint();
//
//   while (true) {
//     llvm::outs() << "parseXXX1: " << Token2String(getToken().getKind()) <<
//     "\n"; if (check(TokenKind::Eof)) {
//       // abort
//       return createStringError(inconvertibleErrorCode(),
//                                "failed to parse "
//                                "PathInExpressionOrStructExprStructOrStructTuple"
//                                "UnitOrMacroInvocationExpression: eof");
//     } else if (checkSimplePathSegment()) {
//       assert(eatSimplePathSegment());
//     } else if (check(TokenKind::Lt)) {
//       recover(cp);
//       return parsePathInExpressionOrStructOrExpressionWithPostfix();
//     } else if (checkPathIdentSegment()) {
//       recover(cp);
//       return parsePathInExpressionOrStructOrExpressionWithPostfix();
//     } else if (check(TokenKind::Not)) {
//       recover(cp);
//       return parseMacroInvocationExpression();
//     } else if (check(TokenKind::BraceOpen)) {
//       recover(cp);
//       return parseStructExpression();
//     } else if (check(TokenKind::ParenOpen)) {
//       recover(cp);
//       return parseStructExpression();
//     } else if (checkPostFix()) {
//       recover(cp);
//       xxx;
//       endless loop return parseExpressionWithPostfix();
//     } else if (check(TokenKind::PathSep)) {
//       assert(eat(TokenKind::PathSep));
//     }
//   }
// }

// StringResult<std::shared_ptr<ast::Expression>>
// Parser::parsePathInExpressionOrStructOrExpressionWithPostfix() {
//   ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
//   CheckPoint cp = getCheckPoint();
//
//   while (true) {
//     llvm::outs() << "parseXXX2: " << Token2String(getToken().getKind()) <<
//     "\n"; if (check(TokenKind::Eof)) {
//       // abort
//       return StringResult<std::shared_ptr<ast::Expression>>(
//           "failed to parse PathInExpressionOrStructOrExpressionWithPostfix: "
//           "eof");
//     } else if (check(TokenKind::BraceOpen)) {
//       recover(cp);
//       return parseStructExpression();
//     } else if (check(TokenKind::ParenOpen)) {
//       recover(cp);
//       return parseStructExpression();
//     } else if (check(TokenKind::PathSep)) {
//       assert(eat(TokenKind::PathSep));
//       //    } else if (check(TokenKind::Lt)) {
//       //      xxx;
//     } else if (checkPostFix()) {
//       recover(cp);
//       return parseExpressionWithPostfix();
//     }
//   }
// }

adt::StringResult<std::shared_ptr<ast::Expression>>
Parser::parseMacroInvocationExpressionPratt(
    std::shared_ptr<ast::Expression> path, std::span<ast::OuterAttribute>,
    Restrictions) {

  MacroInvocationExpression macro = {path->getLocation()};
  macro.setPath(path);

  if (!check(TokenKind::Not))
    return adt::StringResult<std::shared_ptr<ast::Expression>>(
        "failed to parse ! token in parse macro invocation expression pratt");
  assert(eat(TokenKind::Not));

  StringResult<std::shared_ptr<ast::DelimTokenTree>> token =
      parseDelimTokenTree();
  if (!token) {
    llvm::errs()
        << "failed to delim token tree in macro invocation expression: "
        << token.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  macro.setTree(token.getValue());

  return adt::StringResult<std::shared_ptr<ast::Expression>>(
      std::make_shared<MacroInvocationExpression>(macro));
}

} // namespace rust_compiler::parser
