#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <cstdlib>
#include <llvm/Support/raw_ostream.h>

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
      llvm::errs() << "unknown token: " << Token2String(getToken().getKind())
                   << "\n";
      exit(EXIT_FAILURE);
    }
    }
  }
}

} // namespace rust_compiler::parser
