#include "Statements.h"

#include "AST/Statements.h"

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Statements>>
tryParseStatements(std::span<lexer::Token> tokens) {
  Statements stmts = {tokens.front().getLocation()};

  std::span<lexer::Token> view = tokens;

  llvm::errs() << "tryParseStatements"
               << "\n";

  if (view.front().getKind() == TokenKind::Semi) {
    return std::make_shared<ast::Statements>(stmts);
  }

  std::optional<std::shared_ptr<ast::Expression>> woBlock =
      tryParseExpressionWithoutBlock(view);

  if (woBlock) {
    stmts.setTrailing(*woBlock);
    return std::make_shared<ast::Statements>(stmts);
  }

  while (view.size() > 1) {
    std::optional<std::shared_ptr<ast::Statement>> stmt =
        tryParseStatement(view);
    if (stmt) {
      stmts.addStmt(*stmt);
      view = view.subspan((*stmt)->getTokens());
    }
    std::optional<std::shared_ptr<ast::Expression>> woBlock =
        tryParseExpressionWithoutBlock(view);

    if (woBlock) {
      stmts.setTrailing(*woBlock);
      return std::make_shared<ast::Statements>(stmts);
    }
  }

  return std::make_shared<ast::Statements>(stmts);

  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
