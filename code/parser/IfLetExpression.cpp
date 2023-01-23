#include "AST/Patterns/Patterns.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <memory>

using namespace rust_compiler::ast;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
Parser::tryParseIfLetExpression(std::span<lexer::Token> tokens) {

  std::span<lexer::Token> view = tokens;

  if (view.front().isKeyWord() && view.front().getIdentifier() == "if") {
    view = view.subspan(1);
    if (view.front().isKeyWord() && view.front().getIdentifier() == "let") {
      view = view.subspan(1);

      std::optional<std::shared_ptr<ast::patterns::Pattern>> pattern =
          tryParsePattern(view);
      if (pattern) {
        view = view.subspan((*pattern)->getTokens());

        if (view.front().getKind() == lexer::TokenKind::Eq) {
          view = view.subspan(1);

          std::optional<std::shared_ptr<ast::Scrutinee>> scrutinee =
              tryParseScrutinee(view);
          if (scrutinee) {
            view = view.subspan((*scrutinee)->getTokens());
            std::optional<std::shared_ptr<ast::Expression>> block =
                tryParseBlockExpression(view);
            if (block) {
              view = view.subspan((*block)->getTokens());
              IfLetExpression ifLetExpr = {tokens.front().getLocation()};
              ifLetExpr.setPattern(*pattern);
              ifLetExpr.setScrutinee(*scrutinee);
              ifLetExpr.setBlock(*block);

              if (view.front().isKeyWord() &&
                  view.front().getIdentifier() == "else") {
                view = view.subspan(1);
                std::optional<std::shared_ptr<ast::Expression>> block =
                    tryParseBlockExpression(view);
                if (block) {
                  ifLetExpr.setTrailing(*block);
                  return std::static_pointer_cast<ast::Expression>(
                      std::make_shared<IfLetExpression>(ifLetExpr));
                }

                std::optional<std::shared_ptr<ast::Expression>> ifNestedExpr =
                    tryParseIfExpression(view);
                if (ifNestedExpr) {
                  ifLetExpr.setTrailing(*ifNestedExpr);
                  return std::static_pointer_cast<ast::Expression>(
                      std::make_shared<IfLetExpression>(ifLetExpr));
                }
                std::optional<std::shared_ptr<ast::Expression>> ifLetInnerExpr =
                    tryParseIfLetExpression(view);
                if (ifLetInnerExpr) {
                  ifLetExpr.setTrailing(*ifLetInnerExpr);
                  return std::static_pointer_cast<ast::Expression>(
                      std::make_shared<IfLetExpression>(ifLetExpr));
                }
              } else {
                return std::static_pointer_cast<ast::Expression>(
                    std::make_shared<IfLetExpression>(ifLetExpr));
              }
            }
          }
        }
      }
    }
  }
  return std::nullopt;
}

} // namespace rust_compiler::parser
