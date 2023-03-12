#include "AST/Expression.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <cstdlib>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>

using namespace rust_compiler::lexer;
using namespace rust_compiler::adt;

namespace rust_compiler::parser {

adt::Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseExpressionWithoutBlock(std::span<ast::OuterAttribute> outer,
                                    Restrictions restrictions) {
  if (getToken().isKeyWord()) {
    switch (getToken().getKeyWordKind()) {
    case KeyWordKind::KW_RETURN: {
      return parseReturnExpression(outer);
    }
    case KeyWordKind::KW_BREAK: {
      return parseReturnExpression(outer);
    }
    case KeyWordKind::KW_CONTINUE: {
      return parseContinueExpression(outer);
    }
    case KeyWordKind::KW_MOVE: {
      return parseClosureExpression(outer);
    }
    case KeyWordKind::KW_TRUE:
    case KeyWordKind::KW_FALSE: {
      return parseLiteralExpression(outer);
    }
    case KeyWordKind::KW_SUPER:
    case KeyWordKind::KW_SELFVALUE:
    case KeyWordKind::KW_SELFTYPE: {
      return parsePathInExpression();
    }
    default:
      if (auto kw = KeyWord2String(getToken().getKeyWordKind())) {
        llvm::errs() << "parseExpressionWithoutBlock: unknown keyword: " << *kw
                     << "\n";
      } else {
        llvm::errs() << "parseExpressionWithoutBlock : unknown keyword: "
                     << "\n";
      }
      exit(EXIT_FAILURE);
    }
  } else {
    switch (getToken().getKind()) {
    case TokenKind::Lt: {
      return parseQualifiedPathInExpression();
    }
    case TokenKind::SquareOpen: {
      return parseArrayExpression(outer);
    }
    case TokenKind::ParenOpen: {
      return parseGroupedOrTupleExpression(restrictions);
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
    case TokenKind::Or:
    case TokenKind::OrOr:
      // closure expression
      return StringResult<std::shared_ptr<ast::Expression>>(
          parseClosureExpression(outer));
    case TokenKind::Minus: {
      Restrictions enteredFromUnary;
      enteredFromUnary.enteredFromUnary = true;
      if (!restrictions.canBeStructExpr)
        enteredFromUnary.canBeStructExpr = false;
      return StringResult<std::shared_ptr<ast::Expression>>(
          parseExpression(Precedence::UnaryMinus, {}, enteredFromUnary));
    }
    default:
      adt::StringResult<std::shared_ptr<ast::Expression>> expr =
          parseExpression(outer, restrictions);
      if (!expr) {
        llvm::errs() << "parseExpressionWithoutBlock: failed to parse "
                        "expression: "
                     << expr.getError() << "\n";
        std::string s =
            llvm::formatv("{0} {1}",
                          "parseExpressionWithoutBlock: failed to parse "
                          "expression ",
                          expr.getError())
                .str();
        return adt::Result<std::shared_ptr<ast::Expression>, std::string>(s);
      }

      switch (expr.getValue()->getExpressionKind()) {
      case ast::ExpressionKind::ExpressionWithBlock: {
        llvm::errs() << "parseExpressionWithoutBlock: expected expression "
                        "without block but:"
                     << "\n";
        std::string s =
            llvm::formatv(
                "{0} {1}",
                "parseExpressionWithoutBlock: expected expression "
                "without block but:",
                ExpressionWithBlockKind2String(
                    std::static_pointer_cast<ast::ExpressionWithBlock>(
                        expr.getValue())
                        ->getWithBlockKind()))
                .str();
        return adt::Result<std::shared_ptr<ast::Expression>, std::string>(s);
      }
      case ast::ExpressionKind::ExpressionWithoutBlock: {
        return expr;
      }
      }

      // llvm::errs() << "parseExpressionWithoutBlock: unknown token: "
      //              << Token2String(getToken().getKind()) << "\n";
      // exit(EXIT_FAILURE);
      // std::string s =
      //     llvm::formatv("{0} {1}",
      //                   "parseExpressionWithoutBlock: unknown token: ",
      //                   Token2String(getToken().getKind()))
      //         .str();
      // return adt::Result<std::shared_ptr<ast::Expression>, std::string>(s);
    }
    }
  }
}

} // namespace rust_compiler::parser
