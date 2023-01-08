#include "ReturnExpression.h"

#include "AST/ReturnExpression.h"
#include "Expression.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Util.h"

#include <memory>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParseReturnExpression(std::span<Token> tokens) {
  std::span<lexer::Token> view = tokens;

  printTokenState(view);

  if (view.front().isKeyWord() and
      view.front().getKeyWordKind() == KeyWordKind::KW_RETURN) {
    view = view.subspan(1);

    printf("tryParseReturnExpression: found return\n");

    std::optional<std::shared_ptr<ast::Expression>> expr =
        tryParseExpression(view);

    if (expr) {
      auto foo = std::make_shared<ReturnExpression>(
          tokens.front().getLocation(), *expr);
      return std::static_pointer_cast<Expression>(foo);
    } else {
      auto foo = std::make_shared<ReturnExpression>(view.front().getLocation());
      return std::static_pointer_cast<Expression>(foo);
    }
  }

  printf("tryParseReturnExpression: failed: %lu\n", tokens.size());
  if (view[0].isIdentifier()) {
    printf("%s s\n", view[0].getIdentifier().c_str());
  } else {
    printf("%s x%sx\n", Token2String(view[0].getKind()).c_str(),
           view[0].getIdentifier().c_str());
    if (isKeyWord(view[0].getIdentifier())) {
      if (KeyWord2String(view.front().getKeyWordKind())) {
        printf("found keyword: %s\n",
               (*KeyWord2String(view.front().getKeyWordKind())).c_str());
      }
    }
  }

  return std::nullopt;
}

} // namespace rust_compiler::parser