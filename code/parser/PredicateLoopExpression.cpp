#include "AST/PredicateLoopExpression.h"

#include "Parser/Parser.h"

#include <memory>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
Parser::tryParsePredicateLoopExpression(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  if (view.front().isKeyWord() &&
      view.front().getKeyWordKind() == KeyWordKind::KW_WHILE) {
    view = view.subspan(1);

    std::optional<std::shared_ptr<ast::Expression>> condition =
        tryParseExpression(view);
    if (condition) {
      view = view.subspan((*condition)->getTokens());

      std::optional<std::shared_ptr<ast::BlockExpression>> block =
          tryParseBlockExpression(view);

      if (block) {

        PredicateLoopExpression loop = {tokens.front().getLocation()};
        loop.setBody(*block);
        loop.setCondition(*condition);
        return std::static_pointer_cast<Expression>(
            std::make_shared<PredicateLoopExpression>(loop));
      }
    }
  }

  return std::nullopt;
}

} // namespace rust_compiler::parser
