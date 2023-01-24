#include "AST/AssignmentExpression.h"

#include "Parser/Parser.h"

#include <memory>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
Parser::tryParseAssignmentExpression(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  std::optional<std::shared_ptr<ast::Expression>> left =
      tryParseExpression(view);

  if (left) {
    view = view.subspan((*left)->getTokens());

    if (view.front().getKind() == TokenKind::Eq) {
      view = view.subspan(1);

      std::optional<std::shared_ptr<ast::Expression>> right =
          tryParseExpression(view);

      if (right) {
        AssignmentExpression assign = {tokens.front().getLocation()};
        assign.setLeft(*left);
        assign.setRight(*right);
        return std::static_pointer_cast<Expression>(
            std::make_shared<AssignmentExpression>(assign));
      }
    }
  }
  return std::nullopt;
}

} // namespace rust_compiler::ast
