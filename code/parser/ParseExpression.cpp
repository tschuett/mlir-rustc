#include "ADT/Result.h"
#include "AST/ErrorPropagationExpression.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <cstdlib>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#include <span>

using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace rust_compiler::lexer;
using namespace llvm;

namespace rust_compiler::parser {

StringResult<std::shared_ptr<ast::Expression>>
Parser::parseExpression(std::span<ast::OuterAttribute> outer,
                        Restrictions restrictions) {
  return parseExpression(Precedence::Lowest, outer, restrictions);
}

adt::StringResult<std::shared_ptr<ast::Expression>>
Parser::parseExpression(Precedence rightBindingPower,
                        std::span<ast::OuterAttribute> outer,
                        Restrictions restrictions) {
  //  llvm::errs() << "parseExpression"
  //               << "\n";

  adt::StringResult<std::shared_ptr<ast::Expression>> expr =
      parseUnaryExpression({}, restrictions);
  if (!expr) {
    llvm::errs() << "failed to parse unary expression in expression: "
                 << expr.getError() << "\n";
    printFunctionStack();
    std::string s =
        llvm::formatv(
            "{0} {1}",
            "failed to parse unary expression in expression: ", expr.getError())
            .str();
    return adt::StringResult<std::shared_ptr<ast::Expression>>(s);
  }

  llvm::errs() << "token found after unary expression: "
               << Token2String(getToken().getKind()) << "\n";

  // stop parsing if find lower priority token - parse higher priority first
  while (rightBindingPower < getLeftBindingPower()) {
    CheckPoint cp = getCheckPoint();
    expr = parseInfixExpression(expr.getValue(), outer, restrictions);
    if (!expr) {
      llvm::errs() << "failed to parse unary expression in expression: "
                   << expr.getError() << "\n";
      llvm::errs() << "returning nullptr"
                   << "\n";
      recover(cp);
      std::string s =
          llvm::formatv("{0} {1}",
                        "failed to parse unary expression in expression: ",
                        expr.getError())
              .str();

      return adt::StringResult<std::shared_ptr<ast::Expression>>(s);
    }
  }

  return expr;
}

StringResult<std::shared_ptr<ast::Expression>>
Parser::parseInfixExpression(std::shared_ptr<Expression> left,
                             std::span<OuterAttribute> outer,
                             Restrictions restrictions) {
  switch (getToken().getKind()) {
  case TokenKind::QMark: {
    Location loc = getLocation();
    ErrorPropagationExpression error = {loc};
    error.setLhs(left);
    assert(eat(TokenKind::QMark));
    return adt::StringResult<std::shared_ptr<ast::Expression>>(
        std::make_shared<ErrorPropagationExpression>(error));
  }
  case TokenKind::Plus: {
    return parseArithmeticOrLogicalExpression(left, restrictions);
  }
  case TokenKind::Minus: {
    return parseArithmeticOrLogicalExpression(left, restrictions);
  }
  case TokenKind::Lt: {
    return parseComparisonExpression(left, restrictions);
  }
  case TokenKind::Eq: {
    return parseAssignmentExpression(left, outer, restrictions);
  }
  case TokenKind::DotDot: {
    return parseRangeExpression(left, outer, restrictions);
  }
  case TokenKind::DotDotEq: {
    return parseRangeExpression(left, outer, restrictions);
  }
  case TokenKind::SquareOpen: {
    return parseIndexExpression(left, outer, restrictions);
  }
  case TokenKind::ParenOpen: {
    return parseCallExpression(left, outer, restrictions);
  }
  case TokenKind::Dot: {
    // field expression or method call
    if (checkKeyWord(KeyWordKind::KW_AWAIT, 1)) {
      return parseAwaitExpression(left, outer);
    } else if (check(TokenKind::INTEGER_LITERAL, 1)) {
      return parseTupleIndexingExpression(left, outer, restrictions);
    } else if (check(TokenKind::Identifier, 1) &&
               !check(TokenKind::ParenOpen, 2) && !check(TokenKind::PathSep, 2)) {
      return parseFieldExpression(left, outer, restrictions);
    }
    return parseMethodCallExpression(left, outer, restrictions);
  }
  case TokenKind::Eof: {
    std::string s =
        llvm::formatv("{0} {1}", "found eof token in parse infix expression",
                      "eof")
            .str();
    return StringResult<std::shared_ptr<ast::Expression>>(s);
  }
  default: {
    std::string s =
        llvm::formatv("{0} {1}",
                      "error unhandled token kind in parse infix expression: ",
                      Token2String(getToken().getKind()))
            .str();
    llvm::errs() << "error unhandled token kind in parse infix expression: "
                 << Token2String(getToken().getKind()) << "\n";
    return StringResult<std::shared_ptr<ast::Expression>>(s);
    // exit(EXIT_FAILURE);
  }
  }
}

} // namespace rust_compiler::parser

/*
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

*/
