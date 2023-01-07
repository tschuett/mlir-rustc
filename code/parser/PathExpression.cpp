#include "PathExpression.h"

#include "AST/Expression.h"
#include "PathInExpression.h"

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParsePathExpression(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  std::optional<std::shared_ptr<ast::Expression>> pathIn =
      tryParsePathInExpression(view);

  if (pathIn) {
    return *pathIn;
  }

  std::optional<std::shared_ptr<ast::Expression>> qualPathIn =
      tryParseQualifiedPathInExpression(view);

  if (qualPathIn) {
    return *qualPathIn;
  }

  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
