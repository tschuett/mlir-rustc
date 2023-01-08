#include "ExpressionWithBlock.h"

#include "BlockExpression.h"

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParseExpressionWithBlock(std::span<lexer::Token> tokens) {

  std::span<lexer::Token> view = tokens;

  std::optional<std::shared_ptr<ast::BlockExpression>> block =
      tryParseBlockExpression(view);

  if (block) {
  }

  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
