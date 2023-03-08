#include "ADT/Result.h"
#include "AST/BorrowExpression.h"
#include "AST/DereferenceExpression.h"
#include "AST/LiteralExpression.h"
#include "AST/NegationExpression.h"
#include "AST/PathExprSegment.h"
#include "AST/PathIdentSegment.h"
#include "AST/PathInExpression.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/ErrorStack.h"
#include "Parser/Parser.h"
#include "Parser/Precedence.h"
#include "Parser/Restrictions.h"

#include <llvm/Support/raw_ostream.h>
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

  switch (getToken().getKind()) {
  case TokenKind::Identifier: {

    /* best option: parse as path, then extract identifier, macro,
     * struct/enum, or just path info from it */
    StringResult<std::shared_ptr<PathInExpression>> path =
        parsePathInExpressionPratt();
    if (!path) {
      // FIXME: check
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
                       !canTokenStartType(3);

      /* definitely not a block:
       *  path '{' ident ','
       *  path '{' ident ':' [anything] ','
       *  path '{' ident ':' [not a type]
       * otherwise, assume block expr and thus path */

      if (!restrictions.canBeStructExpr && !notABlock) {
        return StringResult<std::shared_ptr<ast::Expression>>(path.getValue());
      }

      return parseStructExpressionStructPratt(path.getValue(), outer);
    }
    case TokenKind::ParenOpen: {
      if (!restrictions.canBeStructExpr) {
        return StringResult<std::shared_ptr<ast::Expression>>(path.getValue());
      }
      return parseStructExpressionTuplePratt(path.getValue(), outer);
    }
    default: {
      if (path.getValue()->isSingleSegment()) {
        return LiteralExpression();
      }
      return StringResult<std::shared_ptr<ast::Expression>>(path.getValue());
    }
    }
  }
  case TokenKind::Shl:
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
      }
    }
  }
  case TokenKind::AndAnd: {
    // FIXME
  }
  case TokenKind::PathSep: {
    // report error
  }
  case TokenKind::Or:
  case TokenKind::OrOr:
    // closure expression
    return StringResult<std::shared_ptr<ast::Expression>>(
        parseClosureExpressionPratt(outer));
  default: {
    llvm::errs() << "error unhandled token kind: "
                 << Token2String(getToken().getKind()) << "\n";
    exit(EXIT_FAILURE);
  }
  }

  if (getToken().isKeyWord()) {
    switch (getToken().getKeyWordKind()) {
    case KeyWordKind::KW_SELFTYPE:
    case KeyWordKind::KW_SELFVALUE:
    case KeyWordKind::KW_DOLLARCRATE:
    case KeyWordKind::KW_CRATE: {
      xxx;
    }
    case KeyWordKind::KW_MOVE: {
      return StringResult<std::shared_ptr<ast::Expression>>(
          parseClosureExpressionPratt(outer));
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
    default: {
      // report error
    }
    }
  }
}

///  Note that this only parses segment-first paths, not global ones, i.e.
///  there is ::. */
StringResult<std::shared_ptr<PathInExpression>>
Parser::parsePathInExpressionPratt() {

  PathInExpression pathIn = PathInExpression(getLocation());

  PathIdentSegment ident = PathIdentSegment(getLocation());

  if (getToken().isIdentifier()) {
    ident.setIdentifier(getToken().getIdentifier());
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
        llvm::outs() << "unknown keywork: " << *kind << "\n";
      }
      return StringResult<std::shared_ptr<PathInExpression>>(
          "unknown keyword in parsePathInExpressionPratt");
    }
    }
  } else {
    llvm::outs() << "unknown token: " << Token2String(getToken().getKind())
                 << "\n";
    return StringResult<std::shared_ptr<PathInExpression>>(
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
      return StringResult<std::shared_ptr<PathInExpression>>(
          "failed to pars path expr segment");
    }

    pathIn.addDoubleColon();
    pathIn.addSegment(seg.getValue());
  }

  return StringResult<std::shared_ptr<PathInExpression>>(
      std::make_shared<PathInExpression>(pathIn));
}

} // namespace rust_compiler::parser
