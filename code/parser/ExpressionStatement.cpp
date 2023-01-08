#include "ExpressionStatement.h"

#include "ExpressionWithBlock.h"
#include "ExpressionWithoutBlock.h"

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParseExpressionStatement(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  std::optional<std::shared_ptr<ast::Expression>> woBlock =
      tryParseExpressionWithoutBlock(view);

  if (woBlock) {
    return *woBlock;
  }

  // then ;

  std::optional<std::shared_ptr<ast::Expression>> withBlock =
      tryParseExpressionWithBlock(view);

  if (withBlock) {
    return *withBlock;
  }

  return std::nullopt;
}

} // namespace rust_compiler::parser
