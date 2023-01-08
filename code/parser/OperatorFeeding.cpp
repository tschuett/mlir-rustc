#include "OperatorFeeding.h"

#include "LiteralExpression.h"
#include "PathExpression.h"

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParseOperatorFeedingExpression(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  std::optional<std::shared_ptr<ast::Expression>> literal =
      tryParseLiteralExpression(view);

  if (literal)
    return *literal;

  std::optional<std::shared_ptr<ast::Expression>> path =
      tryParsePathExpression(view);

  if (path)
    return *path;

  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
