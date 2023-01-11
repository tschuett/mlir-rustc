#include "NegationExpression.h"

#include "AST/Expression.h"
#include "AST/NegationExpression.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <llvm/Support/raw_ostream.h>
#include <memory>

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
Parser::tryParseNegationExpression(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;
  Location loc = tokens.front().getLocation();

  llvm::errs() << "tryParseNegationExpression: "
               << Token2String(view.front().getKind()) << "\n";

  if (view.front().getKind() == TokenKind::Minus) {
    view = view.subspan(1);

    std::optional<std::shared_ptr<ast::Expression>> expr =
        tryParseExpression(view);

    llvm::errs() << "tryParseNegationExpression: " << expr.has_value() << "\n";

    if (expr) {
      NegationExpression neg = {loc};
      neg.setMinus();
      neg.setRight(*expr);

      return std::static_pointer_cast<ast::Expression>(
          std::make_shared<NegationExpression>(neg));
    }
  } else if (view.front().getKind() == TokenKind::Not) {
    view = view.subspan(1);

    llvm::errs() << "tryParseNegationExpression: Not"
                 << "\n";

    std::optional<std::shared_ptr<ast::Expression>> expr =
        tryParseExpression(view);

    llvm::errs() << "tryParseNegationExpression: " << expr.has_value() << "\n";

    if (expr) {
      NegationExpression neg = {loc};
      neg.setNot();
      neg.setRight(*expr);

      return std::static_pointer_cast<ast::Expression>(
          std::make_shared<NegationExpression>(neg));
    }
  }

  // FIXME

  return std::nullopt;
}

} // namespace rust_compiler::parser
