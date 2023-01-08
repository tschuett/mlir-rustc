#include "ReturnExpression.h"

#include "AST/ReturnExpression.h"
#include "Expression.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"

#include <memory>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParseReturnExpression(std::span<Token> tokens) {
  std::span<lexer::Token> view = tokens;

  if (view.front().isKeyWord() and
      view.front().getKeyWordKind() == KeyWordKind::KW_RETURN) {
    view = view.subspan(1);
    std::optional<std::shared_ptr<ast::Expression>> expr =
        tryParseExpression(view);

    if (expr) {
      auto foo = std::make_shared<ReturnExpression>(
          tokens.front().getLocation(), *expr);
      return std::static_pointer_cast<Expression>(foo);
    } else {
      auto foo = std::make_shared<ReturnExpression>(
          tokens.front().getLocation());
      return std::static_pointer_cast<Expression>(foo);
    }
  }

  return std::nullopt;
}

} // namespace rust_compiler::parser
