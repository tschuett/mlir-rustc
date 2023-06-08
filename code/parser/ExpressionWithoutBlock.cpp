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
//  llvm::errs() << "parseExpressionWithoutBlock"
//               << "\n";
//
//  llvm::errs() << Token2String(getToken().getKind()) << "\n";
//  llvm::errs() << getToken().isKeyWord() << "\n";

  bool isKeyWord = getToken().isKeyWord();

  if (isKeyWord) {
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
    default: {
    }
    }
  }
  // normal Token
  switch (getToken().getKind()) {
//  case TokenKind::Lt: {
//    return parseQualifiedPathInExpression();
//  }
//  case TokenKind::SquareOpen: {
//    return parseArrayExpression(outer);
//  }
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
          llvm::formatv("{0} {1}",
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
  }
}

} // namespace rust_compiler::parser
