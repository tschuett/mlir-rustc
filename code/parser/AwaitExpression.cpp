#include "AST/AwaitExpression.h"

#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <memory>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
Parser::tryParseAwaitExpression(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  std::optional<std::shared_ptr<ast::Expression>> lhs =
      tryParseExpression(view);
  if (lhs) {

    if (view.front().getKind() == TokenKind::Dot) {
      view = view.subspan(1);
      if (view.front().isKeyWord() &&
          view.front().getKeyWordKind() == KeyWordKind::KW_AWAIT) {
        view = view.subspan(1);

        return std::static_pointer_cast<Expression>(
            std::make_shared<AwaitExpression>(tokens.front().getLocation(),
                                              *lhs));
      }
    }
  }

  return std::nullopt;
}

} // namespace rust_compiler::parser
