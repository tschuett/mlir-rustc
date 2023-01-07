#include "BlockExpression.h"

#include "Lexer/Token.h"
#include "Statement.h"

#include <optional>

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::BlockExpression>>
tryParseBlockExpression(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  if (view.front().getKind() != TokenKind::BraceOpen)
    return std::nullopt;

  view = view.subspan(1);

  std::optional<std::shared_ptr<ast::Statement>> stmt = tryParseStatement(view);

  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
