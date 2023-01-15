#include "AST/BorrowExpression.h"

#include "Lexer/KeyWords.h"
#include "Parser/Parser.h"

#include <memory>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
Parser::tryParseBorrowExpression(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  if (view.front().getKind() == TokenKind::And or
      view.front().getKind() == TokenKind::AndAnd) {
    view = view.subspan(1);
    if (view.front().isKeyWord() and
        view.front().getKeyWordKind() == KeyWordKind::KW_MUT) {
      view = view.subspan(1);

      std::optional<std::shared_ptr<ast::Expression>> expr =
          tryParseExpression(view);
      if (expr) {
        BorrowExpression borrow = {tokens.front().getLocation()};
        borrow.setExpression(*expr);
        borrow.setMut();
        return std::static_pointer_cast<Expression>(
            std::make_shared<BorrowExpression>(borrow));
      }

    } else {
      std::optional<std::shared_ptr<ast::Expression>> expr =
          tryParseExpression(view);
      if (expr) {
        BorrowExpression borrow = {tokens.front().getLocation()};
        borrow.setExpression(*expr);
        return std::static_pointer_cast<Expression>(
            std::make_shared<BorrowExpression>(borrow));
      }
    }
  } else {
    return std::nullopt;
  }

  return std::nullopt;
}

} // namespace rust_compiler::parser
