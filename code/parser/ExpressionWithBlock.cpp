#include "ExpressionWithBlock.h"

#include "AST/Expression.h"
#include "BlockExpression.h"
#include "Parser/Parser.h"

#include <memory>

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
Parser::tryParseExpressionWithBlock(std::span<lexer::Token> tokens) {

  std::span<lexer::Token> view = tokens;

  std::optional<std::shared_ptr<ast::BlockExpression>> block =
      tryParseBlockExpression(view);

  if (block) {
    return std::static_pointer_cast<ast::Expression>(*block);
  }

  std::optional<std::shared_ptr<ast::Expression>> ifExpr =
      tryParseIfExpression(view);
  if (ifExpr) {
    return *ifExpr;
  }

    std::optional<std::shared_ptr<ast::Expression>> ifLetExpr =
      tryParseIfLetExpression(view);
  if (ifLetExpr) {
    return *ifLetExpr;
  }

  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
