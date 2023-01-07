#include "LiteralExpression.h"

#include "Lexer/Token.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParseLiteralExpression(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  if (view.front().getKind() == TokenKind::DecIntegerLiteral) {
  }

  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
