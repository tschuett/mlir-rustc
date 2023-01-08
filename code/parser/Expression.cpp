#include "Expression.h"

using namespace rust_compiler::ast;
using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParseExpression(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  std::optional<std::shared_ptr<ast::Expression>> woBlock =
      tryParseExpressionWithoutBlock(view);
  if (woBlock) {
    return woBlock;
  }

  std::optional<std::shared_ptr<ast::Expression>> withBlock =
      tryParseExpressionWithBlock(view);
  if (withBlock) {
    return withBlock;
  }

  return std::nullopt;
}

} // namespace rust_compiler::parser