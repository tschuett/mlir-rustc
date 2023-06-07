#include "AST/OuterAttribute.h"
#include "Parser/Parser.h"

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::adt;
using namespace rust_compiler::ast;
using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

StringResult<std::shared_ptr<ast::Expression>>
Parser::parseRangeExpression(std::span<OuterAttribute> outer) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  RangeExpression range = {loc};

  llvm::errs() << "parseRangeExpression"
               << "\n";

  if (check(TokenKind::DotDot)) {
    assert(eat(TokenKind::DotDot));
    if (checkRangeTerminator()) {
      // done
      range.setKind(RangeExpressionKind::RangeFullExpr);
      return StringResult<std::shared_ptr<ast::Expression>>(
          std::make_shared<RangeExpression>(range));
    } else {
      // parse
      Restrictions restrictions;
      StringResult<std::shared_ptr<ast::Expression>> right =
          parseExpression(Precedence::DotDot, {}, restrictions);
      if (!right) {
        llvm::errs() << "failed to parse epxression in range expression: "
                     << right.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      range.setRight(right.getValue());
      range.setKind(RangeExpressionKind::RangeToExpr);
      return StringResult<std::shared_ptr<ast::Expression>>(
          std::make_shared<RangeExpression>(range));
    }
  } else if (check(TokenKind::DotDotEq)) {
    assert(eat(TokenKind::DotDotEq));
    // parse
    Restrictions restrictions;
    StringResult<std::shared_ptr<ast::Expression>> right =
        parseExpression(Precedence::DotDotEq, {}, restrictions);
    if (!right) {
      llvm::errs() << "failed to parse expression in range expression: "
                   << right.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    range.setRight(right.getValue());
    range.setKind(RangeExpressionKind::RangeToInclusiveExpr);
    return StringResult<std::shared_ptr<ast::Expression>>(
        std::make_shared<RangeExpression>(range));
  }

  //  Restrictions restrictions;
  //  StringResult<std::shared_ptr<ast::Expression>> left =
  //      parseExpression({}, restrictions);
  //  if (!left) {
  //    llvm::errs() << "failed to parse expression in range expression: "
  //                 << left.getError() << "\n";
  //    printFunctionStack();
  //    exit(EXIT_FAILURE);
  //  }
  //
  //  return parseRangeExpression(left.getValue());
  //
  //

  llvm::errs() << "failed to parse range expression"
               << "\n";
  return StringResult<std::shared_ptr<ast::Expression>>(
      "failed to parse range expression");
}

StringResult<std::shared_ptr<ast::Expression>>
Parser::parseRangeExpression(std::shared_ptr<ast::Expression> l,
                             std::span<OuterAttribute> outer,
                             Restrictions restrictions) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  RangeExpression range = {loc};

//  llvm::errs() << "parseRangeExpression2"
//               << "\n";

  range.setLeft(l);

  if (check(TokenKind::DotDot)) {
    assert(eat(TokenKind::DotDot));
    if (checkRangeTerminator()) {
      range.setKind(RangeExpressionKind::RangeFromExpr);
      return StringResult<std::shared_ptr<ast::Expression>>(
          std::make_shared<RangeExpression>(range));
    } else {
      Restrictions restrictions;
      StringResult<std::shared_ptr<ast::Expression>> left =
          parseExpression({}, restrictions);
      if (!left) {
        llvm::errs() << "failed to parse expression in range expression: "
                     << left.getError() << "\n";
        printFunctionStack();
        std::string s =
            llvm::formatv("{0}\n{1}",
                          "failed to parse expression in range expression: ",
                          left.getError())
                .str();
        return StringResult<std::shared_ptr<ast::Expression>>(s);
      }
      range.setRight(left.getValue());
    }
    range.setKind(RangeExpressionKind::RangeExpr);
    return StringResult<std::shared_ptr<ast::Expression>>(
        std::make_shared<RangeExpression>(range));
  } else if (check(TokenKind::DotDotEq)) {
    assert(eat(TokenKind::DotDotEq));
    range.setKind(RangeExpressionKind::RangeInclusiveExpr);
    Restrictions restrictions;
    Result<std::shared_ptr<ast::Expression>, std::string> left =
        parseExpression({}, restrictions);
    if (!left) {
      llvm::errs() << "failed to parse expression in range expression: "
                   << left.getError() << "\n";
      printFunctionStack();
      // exit(EXIT_FAILURE);
      std::string s =
          llvm::formatv("{0}\n{1}",
                        "failed to parse expression in range expression: ",
                        left.getError())
              .str();
      return StringResult<std::shared_ptr<ast::Expression>>(s);
    }
    range.setRight(left.getValue());
    return StringResult<std::shared_ptr<ast::Expression>>(
        std::make_shared<RangeExpression>(range));
  }
  return StringResult<std::shared_ptr<ast::Expression>>(
      "failed to range expression");
}

} // namespace rust_compiler::parser
