#include "BlockExpression.h"

#include "AST/BlockExpression.h"
#include "Lexer/Token.h"
#include "Statement.h"
#include "Statements.h"

#include <llvm/Support/raw_ostream.h>
#include <optional>

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::BlockExpression>>
tryParseBlockExpression(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;
  BlockExpression block = {tokens.front().getLocation()};

  llvm::errs() << "tryParseBlockExpression"
               << "\n";

  if (view.front().getKind() != TokenKind::BraceOpen)
    return std::nullopt;

  view = view.subspan(1);

  if (view.front().getKind() == TokenKind::BraceClose) {
    return std::make_shared<ast::BlockExpression>(block);
  }

  std::optional<std::shared_ptr<ast::Statements>> stmts =
      tryParseStatements(view);
  if (stmts) {
    view = view.subspan((*stmts)->getTokens());
    if (view.front().getKind() == TokenKind::BraceClose) {
      block.setStatements(*stmts);
      return std::make_shared<ast::BlockExpression>(block);
    }
  }

  return std::nullopt;
}

} // namespace rust_compiler::parser

// FIXME
