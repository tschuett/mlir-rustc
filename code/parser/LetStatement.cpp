#include "AST/LetStatement.h"

#include "AST/Expression.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Statement.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/Types.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <optional>

using namespace rust_compiler::ast;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Statement>>
Parser::tryParseLetStatement(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  // OuterAttributes
  // Else Block

  llvm::errs() << "tryParseLetStatement"
               << "\n";

  LetStatement let = {tokens.front().getLocation()};

  if (view.front().isKeyWord() and
      view.front().getKeyWordKind() == lexer::KeyWordKind::KW_LET) {
    view = view.subspan(1);
    std::optional<std::shared_ptr<ast::patterns::PatternNoTopAlt>> pat =
        tryParsePatternNoTopAlt(view);
    if (pat) {
      view = view.subspan((*pat)->getTokens());
      let.setPattern(*pat);
      // type
      if (view.front().getKind() == lexer::TokenKind::Colon) {
        view = view.subspan(1);
        std::optional<std::shared_ptr<ast::types::TypeExpression>> type =
            tryParseTypeExpression(view);
        if (type) {
          view = view.subspan((*type)->getTokens());
          let.setType(*type);
        }
      }
      // expression
      if (view.front().getKind() == lexer::TokenKind::Eq) {
        view = view.subspan(1);
        std::optional<std::shared_ptr<ast::Expression>> expr =
            tryParseExpression(view);
        if (expr) {
          view = view.subspan((*expr)->getTokens());
          let.setExpression(*expr);
        }
      }
    } else { // No Pat
      return std::nullopt;
    }
    if (view.front().getKind() == lexer::TokenKind::Semi) {
      return std::static_pointer_cast<ast::Statement>(
          std::make_shared<LetStatement>(let));
    }
  }

  return std::nullopt;
}

} // namespace rust_compiler::parser
