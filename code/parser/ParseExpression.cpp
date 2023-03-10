#include "ADT/Result.h"
#include "AST/ErrorPropagationExpression.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <cstdlib>
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
  CheckPoint cp = getCheckPoint();

//  llvm::outs() << "parseExpression"
//               << "\n";
//
  adt::StringResult<std::shared_ptr<ast::Expression>> expr =
      parseUnaryExpression({}, restrictions);
  if (!expr) {
    llvm::errs() << "failed to parse unary expression in expression: "
                 << expr.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }

  // stop parsing if find lower priority token - parse higher priority first
  while (rightBindingPower < getLeftBindingPower(getToken())) {
    CheckPoint cp = getCheckPoint();
    expr = parseInfixExpression(expr.getValue(), outer, restrictions);
    if (!expr) {
      llvm::errs() << "failed to parse unary expression in expression: "
                   << expr.getError() << "\n";
      llvm::errs() << "returning nullptr"
                   << "\n";
      recover(cp);
      return adt::StringResult<std::shared_ptr<ast::Expression>>("");
    }
  }

  return expr;
}

StringResult<std::shared_ptr<ast::Expression>>
Parser::parseInfixExpression(std::shared_ptr<Expression> left,
                             std::span<OuterAttribute>,
                             Restrictions restrictions) {
  switch (getToken().getKind()) {
  case TokenKind::QMark: {
    Location loc = getLocation();
    ErrorPropagationExpression error = {loc};
    error.setLhs(left);
    return adt::StringResult<std::shared_ptr<ast::Expression>>(
        std::make_shared<ErrorPropagationExpression>(error));
  }
  case TokenKind::Plus: {
    return parseArithmeticOrLogicalExpression(left, restrictions);
  }
  case TokenKind::Minus: {
    return parseArithmeticOrLogicalExpression(left, restrictions);
  }
  default: {
    llvm::errs() << "error unhandled token kind: "
                 << Token2String(getToken().getKind()) << "\n";
    exit(EXIT_FAILURE);
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
