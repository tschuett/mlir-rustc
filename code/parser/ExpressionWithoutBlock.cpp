#include "ExpressionWithoutBlock.h"

#include "AST/Expression.h"
#include "LiteralExpression.h"

#include <optional>

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParseExpressionWithoutBlock(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  std::optional<std::shared_ptr<ast::Expression>> lit =
      tryParseLiteralExpression(view);
  if (lit) {
    return *lit;
  }

  std::optional<std::shared_ptr<ast::Expression>> op =
      tryParseOperatorExpression(view);

  if (op) {
    return *op;
  }

  std::optional<std::shared_ptr<ast::Expression>> ret =
      tryParseReturnExpression(view);
  if (ret) {
    return *ret;
  }

  // FIXME

  return std::nullopt;
}

} // namespace rust_compiler::parser
