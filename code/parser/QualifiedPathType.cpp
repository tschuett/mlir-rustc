#include "Lexer/Token.h"
#include "Parser/Parser.h"

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::types::TypeExpression>>
Parser::tryParseQualifiedPathType(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  if (view.front().getKind() == TokenKind::Lt) {
    view = view.subspan(1);

    std::optional<std::shared_ptr<ast::types::TypeExpression>> type =
        tryParseTypeExpression(view);
    if (type) {
      view = view.subspan((*type)->getTokens());
      if (view.front().isKeyWord() && view.front().getIdentifier() == "as") {
        view = view.subspan(1);
        std::optional<std::shared_ptr<ast::types::TypeExpression>> typePath =
            tryParseTypePath(view);
        if (typePath) {
        }
      }
    }
  }

  return std::nullopt;

  assert(false);
}

} // namespace rust_compiler::parser
