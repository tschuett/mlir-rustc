#include "ReturnExpression.h"

#include "AST/ReturnExpression.h"
#include "Expression.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<ast::ReturnExpression>
tryParseReturnExpression(std::span<Token> tokens) {
  std::span<lexer::Token> view = tokens;

  if (view.front().isKeyWord() and
      view.front().getKeyWordKind() == KeyWordKind::KW_RETURN) {
    view = view.subspan(1);
    std::optional<std::shared_ptr<ast::Expression>> expr =
        tryParseExpression(view);

    if (expr) {
      return ReturnExpression(tokens.front().getLocation(), *expr);
    }
  }

  return std::nullopt;
}

} // namespace rust_compiler::parser
