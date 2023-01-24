#include "AST/Statements.h"

#include "Parser/Parser.h"

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Statements>>
Parser::tryParseStatements(std::span<lexer::Token> tokens) {
  Statements stmts = {tokens.front().getLocation()};
  std::span<lexer::Token> view = tokens;

  llvm::errs() << "tryParseStatements: "
               << "\n";

  if (view.front().getKind() == TokenKind::Semi) {
    stmts.setOnlySemi();
    return std::make_shared<ast::Statements>(stmts);
  }

  std::optional<std::shared_ptr<ast::Expression>> woBlock =
      tryParseExpressionWithoutBlock(view);

  if (woBlock) {
    llvm::errs() << "tryParseStatements: found woBlock TRAILING"
                 << "\n";
    stmts.setTrailing(*woBlock);
    return std::make_shared<ast::Statements>(stmts);
  }
  llvm::errs() << "tryParseStatement: while loop"
               << "\n";

  while (view.size() > 1) {
    llvm::errs() << "tryParseStatement: "
                 << "\n";
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
}

} // namespace rust_compiler::parser
