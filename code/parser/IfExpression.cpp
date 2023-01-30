#include "AST/IfExpression.h"
#include "AST/BlockExpression.h"

#include "Parser/Parser.h"

#include <memory>

using namespace rust_compiler::ast;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
Parser::tryParseIfExpression(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  llvm::errs() << "tryParseIfExpression"
               << "\n";

  if (view.front().isKeyWord() && view.front().getIdentifier() == "if") {
    view = view.subspan(1);

    std::optional<std::shared_ptr<ast::Expression>> condition =
        tryParseExpression(view);
    if (condition) {
      view = view.subspan((*condition)->getTokens());
      std::optional<std::shared_ptr<ast::BlockExpression>> block =
          tryParseBlockExpression(view);
      if (block) {
        view = view.subspan((*block)->getTokens());
        IfExpression ifExpr = {tokens.front().getLocation()};
        ifExpr.setCondition(*condition);
        ifExpr.setBlock(*block);
        if (view.front().isKeyWord() &&
            view.front().getIdentifier() == "else") {
          view = view.subspan(1);
          std::optional<std::shared_ptr<ast::Expression>> block =
              tryParseBlockExpression(view);
          if (block) {
            ifExpr.setTrailing(*block);
            return std::static_pointer_cast<ast::Expression>(
                std::make_shared<IfExpression>(ifExpr));
          }

          std::optional<std::shared_ptr<ast::Expression>> ifNestedExpr =
              tryParseIfExpression(view);
          if (ifNestedExpr) {
            ifExpr.setTrailing(*ifNestedExpr);
            return std::static_pointer_cast<ast::Expression>(
                std::make_shared<IfExpression>(ifExpr));
          }
          std::optional<std::shared_ptr<ast::Expression>> ifLetExpr =
              tryParseIfLetExpression(view);
          if (ifLetExpr) {
            ifExpr.setTrailing(*ifLetExpr);
            return std::static_pointer_cast<ast::Expression>(
                std::make_shared<IfExpression>(ifExpr));
          }
        } else {
          return std::static_pointer_cast<ast::Expression>(
              std::make_shared<IfExpression>(ifExpr));
        }
      }
    }
  }

  llvm::errs() << "tryParseIfExpression: nullopt"
               << "\n";

  return std::nullopt;
}

} // namespace rust_compiler::parser
