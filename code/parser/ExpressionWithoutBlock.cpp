#include "AST/Expression.h"
#include "ReturnExpression.h"

#include <optional>

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParseExpressionWithoutBlock(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  std::optional<ast::ReturnExpression> ret = tryParseReturnExpression(view);

  std::optional<std::shared_ptr<ast::Expression>> op=
      tryParseOperatorExpression(view);
  // FIXME

  return std::nullopt;
}

} // namespace rust_compiler::parser
