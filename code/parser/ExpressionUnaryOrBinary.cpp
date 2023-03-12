#include "ADT/Result.h"
#include "AST/BorrowExpression.h"
#include "AST/DereferenceExpression.h"
#include "AST/LiteralExpression.h"
#include "AST/NegationExpression.h"
#include "AST/PathExprSegment.h"
#include "AST/PathExpression.h"
#include "AST/PathIdentSegment.h"
#include "AST/PathInExpression.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/ErrorStack.h"
#include "Parser/Parser.h"
#include "Parser/Precedence.h"
#include "Parser/Restrictions.h"

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <span>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace llvm;

namespace rust_compiler::parser {

// StringResult<std::shared_ptr<ast::Expression>>
// Parser::parseBinaryExpression(std::shared_ptr<ast::Expression> left,
//                               std::span<ast::OuterAttribute> outer,
//                               Restrictions restrictions) {
//   ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
//
//   if (check(TokenKind::QMark)) {
//     return parseErrorPropagationExpression(left, outer);
//   } else if (check(TokenKind::Plus)) {
//     return parseArithmeticOrLogicalExpression(left, restrictions);
//   } else if (check(TokenKind::Minus)) {
//     return parseArithmeticOrLogicalExpression(left, restrictions);
//   }
// }

StringResult<std::shared_ptr<ast::Expression>>
Parser::parseUnaryExpression(std::span<ast::OuterAttribute> outer,
                             Restrictions restrictions) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};

  llvm::errs() << "parseUnaryExpression"
               << "\n";

  Token tok = getToken();

  if (!getToken().isKeyWord()) {

    switch (getToken().getKind()) {
    case TokenKind::Identifier: {

      /* best option: parse as path, then extract identifier, macro,
       * struct/enum, or just path info from it */
      StringResult<std::shared_ptr<Expression>> path =
          parsePathInExpressionPratt();
      if (!path) {
        llvm::errs() << "failed to parse pathin expression pratt: "
                     << path.getError() << "\n";
        return StringResult<std::shared_ptr<ast::Expression>>(
            "failed to parse pathin expression prat");
      }

      switch (getToken().getKind()) {
      case TokenKind::Not: {
        return parseMacroInvocationExpressionPratt(path.getValue(), outer,
                                                   restrictions);
      }
      case TokenKind::BraceOpen: {
        bool notABlock = (getToken(1).isIdentifier() &&
                          getToken(2).getKind() == TokenKind::Comma) ||
                         (getToken(2).getKind() == TokenKind::Colon &&
                          getToken(4).getKind() == TokenKind::Comma) ||
                         !canTokenStartType(getToken(3));

        /* definitely not a block:
         *  path '{' ident ','
         *  path '{' ident ':' [anything] ','
         *  path '{' ident ':' [not a type]
         * otherwise, assume block expr and thus path */

        if (!restrictions.canBeStructExpr && !notABlock) {
          return StringResult<std::shared_ptr<ast::Expression>>(
              path.getValue());
        }

        return parseStructExpressionStructPratt(path.getValue(), outer);
      }
      case TokenKind::ParenOpen: {
        if (!restrictions.canBeStructExpr) {
          return StringResult<std::shared_ptr<ast::Expression>>(
              path.getValue());
        }
        return parseStructExpressionTuplePratt(path.getValue(), outer);
      }
      default: {
        return StringResult<std::shared_ptr<ast::Expression>>(path.getValue());
      }
      }
      break;
    }
    case TokenKind::Lt: {
      return parseQualifiedPathInExpression();
      break;
    }
    case TokenKind::INTEGER_LITERAL:
    case TokenKind::FLOAT_LITERAL:
    case TokenKind::STRING_LITERAL:
    case TokenKind::CHAR_LITERAL:
    case TokenKind::RAW_STRING_LITERAL:
    case TokenKind::BYTE_STRING_LITERAL:
    case TokenKind::RAW_BYTE_STRING_LITERAL:
    case TokenKind::BYTE_LITERAL: {
      return StringResult<std::shared_ptr<ast::Expression>>(
          parseLiteralExpression(outer));
    }
    case TokenKind::ParenOpen: {
      return StringResult<std::shared_ptr<ast::Expression>>(
          parseGroupedOrTupleExpression(restrictions));
    }
    case TokenKind::Minus: {
      Restrictions enteredFromUnary;
      enteredFromUnary.enteredFromUnary = true;
      if (!restrictions.canBeStructExpr)
        enteredFromUnary.canBeStructExpr = false;
      return StringResult<std::shared_ptr<ast::Expression>>(
          parseExpression(Precedence::UnaryMinus, {}, enteredFromUnary));
    }
    case TokenKind::Not: {
      Restrictions enteredFromUnary;
      enteredFromUnary.enteredFromUnary = true;
      if (!restrictions.canBeStructExpr)
        enteredFromUnary.canBeStructExpr = false;
      StringResult<std::shared_ptr<Expression>> expr =
          parseExpression(Precedence::UnaryNot, {}, enteredFromUnary);

      if (expr) {
        NegationExpression neg = {getLocation()};
        neg.setRight(expr.getValue());
        neg.setNot();
        return StringResult<std::shared_ptr<ast::Expression>>(
            std::make_shared<NegationExpression>(neg));
      } else {
        // report error
        return StringResult<std::shared_ptr<ast::Expression>>(
            "failed to parse expression in unary expression with UnaryNot");
      }
    }
    case TokenKind::Star: {
      Restrictions enteredFromUnary;
      enteredFromUnary.enteredFromUnary = true;
      if (!restrictions.canBeStructExpr)
        enteredFromUnary.canBeStructExpr = false;
      StringResult<std::shared_ptr<Expression>> expr =
          parseExpression(Precedence::UnaryStar, {}, enteredFromUnary);
      if (expr) {
        DereferenceExpression der = {getLocation()};
        der.setExpression(expr.getValue());
        return StringResult<std::shared_ptr<ast::Expression>>(
            std::make_shared<DereferenceExpression>(der));
      } else {
        // report error
        return StringResult<std::shared_ptr<ast::Expression>>(
            "failed to parse expression in unary expression with UnaryStar");
      }
    }
    case TokenKind::And: {
      Restrictions enteredFromUnary;
      enteredFromUnary.enteredFromUnary = true;
      enteredFromUnary.canBeStructExpr = false;

      if ((getToken(1).getKind() == TokenKind::Keyword) &&
          (getToken(1).getKeyWordKind() == KeyWordKind::KW_MUT)) {
        assert(eat(TokenKind::Keyword));
        StringResult<std::shared_ptr<Expression>> expr =
            parseExpression(Precedence::UnaryAndMut, {}, enteredFromUnary);
        if (expr) {
          BorrowExpression borrow = {getLocation()};
          borrow.setMut();
          borrow.setExpression(expr.getValue());
          return StringResult<std::shared_ptr<ast::Expression>>(
              std::make_shared<BorrowExpression>(borrow));
        } else {
          // report error
          return StringResult<std::shared_ptr<ast::Expression>>(
              "failed to parse expression in unary expression with "
              "UnaryAndMut");
        }
      } else {
        StringResult<std::shared_ptr<Expression>> expr =
            parseExpression(Precedence::UnaryAnd, {}, enteredFromUnary);
        if (expr) {
          BorrowExpression borrow = {getLocation()};
          borrow.setExpression(expr.getValue());
          return StringResult<std::shared_ptr<ast::Expression>>(
              std::make_shared<BorrowExpression>(borrow));
        } else {
          // report error
          return StringResult<std::shared_ptr<ast::Expression>>(
              "failed to parse expression in unary expression with UnaryAnd");
        }
      }
    }
    case TokenKind::AndAnd: {
      Restrictions enteredFromUnary;
      enteredFromUnary.enteredFromUnary = true;
      enteredFromUnary.canBeStructExpr = false;

      if ((getToken(1).getKind() == TokenKind::Keyword) &&
          (getToken(1).getKeyWordKind() == KeyWordKind::KW_MUT)) {
        assert(eat(TokenKind::Keyword));
        StringResult<std::shared_ptr<Expression>> expr =
            parseExpression(Precedence::UnaryAndMut, {}, enteredFromUnary);
        if (expr) {
          BorrowExpression borrow = {getLocation()};
          borrow.setMut();
          borrow.setExpression(expr.getValue());
          borrow.setDoubleBorrow();
          return StringResult<std::shared_ptr<ast::Expression>>(
              std::make_shared<BorrowExpression>(borrow));
        } else {
          // report error
          return StringResult<std::shared_ptr<ast::Expression>>(
              "failed to parse expression in unary expression with "
              "UnaryAndMut");
        }
      } else {
        StringResult<std::shared_ptr<Expression>> expr =
            parseExpression(Precedence::UnaryAnd, {}, enteredFromUnary);
        if (expr) {
          BorrowExpression borrow = {getLocation()};
          borrow.setExpression(expr.getValue());
          borrow.setDoubleBorrow();
          return StringResult<std::shared_ptr<ast::Expression>>(
              std::make_shared<BorrowExpression>(borrow));
        } else {
          // report error
          return StringResult<std::shared_ptr<ast::Expression>>(
              "failed to parse expression in unary expression with UnaryAnd");
        }
      }
      break;
    }
    case TokenKind::PathSep: {
      // report error
      return StringResult<std::shared_ptr<ast::Expression>>(
          "unexpected :: in unary expression with UnaryAnd");
    }
    case TokenKind::Or:
    case TokenKind::OrOr:
      // closure expression
      return StringResult<std::shared_ptr<ast::Expression>>(
          parseClosureExpression(outer));
    case TokenKind::SquareOpen: {
      return parseArrayExpression(outer);
    }
    default: {
      llvm::errs() << "parseUnaryExpressio2n: error unhandled token kind: "
                   << Token2String(getToken().getKind()) << "\n";
      std::string s =
          llvm::formatv("{0} {1}",
                        "parseUnaryExpression: error unhandled token kind: ",
                        Token2String(getToken().getKind()))
              .str();

      return StringResult<std::shared_ptr<ast::Expression>>(s);
    }
    }
  }

  if (getToken().isKeyWord()) {
    switch (getToken().getKeyWordKind()) {
    case KeyWordKind::KW_SELFTYPE:
    case KeyWordKind::KW_SELFVALUE:
    case KeyWordKind::KW_DOLLARCRATE:
    case KeyWordKind::KW_CRATE: {

      /* best option: parse as path, then extract identifier, macro,
       * struct/enum, or just path info from it */
      StringResult<std::shared_ptr<Expression>> path =
          parsePathInExpressionPratt();
      if (!path) {
        // handle error
        return StringResult<std::shared_ptr<ast::Expression>>(
            "failed to parse pathin expression in unary expression");
      }

      std::shared_ptr<PathInExpression> p =
          std::static_pointer_cast<PathInExpression>(path.getValue());
      if (tok.isKeyWord() &&
          (tok.getKeyWordKind() == KeyWordKind::KW_SELFVALUE) &&
          p->isSingleSegment()) {
        return path;
      }

      switch (getToken().getKind()) {
      case TokenKind::Not: {
        return parseMacroInvocationExpressionPratt(path.getValue(), outer,
                                                   restrictions);
      }
      case TokenKind::BraceOpen: {

        bool notaBlock = (getToken(1).isIdentifier() &&
                          getToken(2).getKind() == TokenKind::Comma) ||
                         (getToken(2).getKind() == TokenKind::Colon &&
                          getToken(4).getKind() == TokenKind::Comma) ||
                         !canTokenStartType(getToken(3));
        if (!restrictions.canBeStructExpr && !notaBlock)
          return path;
        return parseStructExpressionStructPratt(path.getValue(), outer);
      }
      case TokenKind::ParenOpen: {
        if (!restrictions.canBeStructExpr)
          return path;
        return parseStructExpressionTuplePratt(path.getValue(), outer);
      }
      default: {
        return path;
      }
      }
    }
    case KeyWordKind::KW_MOVE: {
      return StringResult<std::shared_ptr<ast::Expression>>(
          parseClosureExpression(outer));
    }
    case KeyWordKind::KW_RETURN: {
      return StringResult<std::shared_ptr<ast::Expression>>(
          parseReturnExpression(outer));
    }
    case KeyWordKind::KW_BREAK: {
      return StringResult<std::shared_ptr<ast::Expression>>(
          parseBreakExpression(outer));
    }
    case KeyWordKind::KW_CONTINUE: {
      return StringResult<std::shared_ptr<ast::Expression>>(
          parseContinueExpression(outer));
    }
    case KeyWordKind::KW_IF: {
      if (getToken(1).isKeyWord() &&
          getToken(1).getKeyWordKind() == KeyWordKind::KW_LET) {
        return StringResult<std::shared_ptr<ast::Expression>>(
            parseIfLetExpression(outer));
      } else {
        return StringResult<std::shared_ptr<ast::Expression>>(
            parseIfExpression(outer));
      }
    }
    case KeyWordKind::KW_LOOP: {
      return StringResult<std::shared_ptr<ast::Expression>>(
          parseLoopExpression(outer));
    }
    case KeyWordKind::KW_WHILE: {
      if (getToken(1).isKeyWord() &&
          getToken(1).getKeyWordKind() == KeyWordKind::KW_LET) {
        return StringResult<std::shared_ptr<ast::Expression>>(
            parsePredicatePatternLoopExpression(outer));
      } else {
        return StringResult<std::shared_ptr<ast::Expression>>(
            parsePredicateLoopExpression(outer));
      }
    }
    case KeyWordKind::KW_MATCH: {
      return StringResult<std::shared_ptr<ast::Expression>>(
          parseMatchExpression(outer));
    }
    case KeyWordKind::KW_TRUE:
    case KeyWordKind::KW_FALSE: {
      return parseLiteralExpression(outer);
    }
    default: {
      llvm::outs() << "unexpected token: " << Token2String(getToken().getKind())
                   << "\n";
      llvm::outs() << "in parseUnaryExpression: "
                   << KeyWord2String(getToken().getKeyWordKind()) << "\n";
      return StringResult<std::shared_ptr<ast::Expression>>(
          "unexpected token in unary expression");
    }
    }
  }
}

///  Note that this only parses segment-first paths, not global ones, i.e.
///  there is ::. */
StringResult<std::shared_ptr<Expression>> Parser::parsePathInExpressionPratt() {

  llvm::outs() << "parse pathin expression pratt"
               << "\n";

  PathInExpression pathIn = PathInExpression(getLocation());

  PathIdentSegment ident = PathIdentSegment(getLocation());

  if (getToken().isIdentifier()) {
    ident.setIdentifier(getToken().getIdentifier());
    assert(eat(TokenKind::Identifier));
  } else if (getToken().isKeyWord()) {
    switch (getToken().getKeyWordKind()) {
    case KeyWordKind::KW_SUPER:
      ident.setSuper();
      break;
    case KeyWordKind::KW_SELFVALUE:
      ident.setSelfValue();
      break;
    case KeyWordKind::KW_SELFTYPE:
      ident.setSelfType();
      break;
    case KeyWordKind::KW_CRATE:
      ident.setCrate();
      break;
    case KeyWordKind::KW_DOLLARCRATE:
      ident.setDollarCrate();
      break;
    default: {
      std::optional<std::string> kind =
          KeyWord2String(getToken().getKeyWordKind());
      if (kind) {
        llvm::outs() << "unknown keyword: " << *kind << "\n";
      }
      return StringResult<std::shared_ptr<Expression>>(
          "unknown keyword in parsePathInExpressionPratt");
    }
    }
    assert(eat(TokenKind::Keyword));
  } else {
    llvm::outs() << "unknown token: " << Token2String(getToken().getKind())
                 << "in start of pathin expression"
                 << "\n";
    return StringResult<std::shared_ptr<Expression>>(
        "unknown token in parsePathInExpressionPratt");
  }

  PathExprSegment initialSegment = PathExprSegment(getLocation());
  initialSegment.addIdentSegment(ident);

  if ((getToken(0).getKind() == TokenKind::PathSep) &&
      (getToken(1).getKind() == TokenKind::Lt)) {
    assert(eat(TokenKind::PathSep));

    adt::Result<ast::GenericArgs, std::string> arg = parseGenericArgs();
    if (arg)
      initialSegment.addGenerics(arg.getValue());
  }

  pathIn.addSegment(initialSegment);

  while (getToken().getKind() == TokenKind::PathSep) {
    assert(eat(TokenKind::PathSep));

    // parse the segment, it is a real error if it fails
    adt::Result<ast::PathExprSegment, std::string> seg = parsePathExprSegment();
    if (!seg) {
      // ignore error ???
      llvm::errs() << "failed to parse expected path expr segment in path in "
                      "expression pratt"
                   << "\n";
      return StringResult<std::shared_ptr<Expression>>(
          "failed to pars path expr segment");
    }

    pathIn.addDoubleColon();
    pathIn.addSegment(seg.getValue());
  }

  return StringResult<std::shared_ptr<Expression>>(
      std::make_shared<PathInExpression>(pathIn));
}

} // namespace rust_compiler::parser
