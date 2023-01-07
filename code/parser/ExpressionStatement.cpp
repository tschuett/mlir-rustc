#include "ExpressionStatement.h"

#include "ExpressionWithoutBlock.h"
#include "ExpressionWithBlock.h"

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParseExpressionStatement(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  std::optional<std::shared_ptr<ast::Expression>> woBlock =
      tryParseExpressionWithoutBlock(view);

  // then ;

  std::optional<std::shared_ptr<ast::Expression>> withBlock =
      tryParseExpressionWithBlock(view);

  return std::nullopt;
}

} // namespace rust_compiler::parser
