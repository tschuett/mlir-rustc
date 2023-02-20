#include "AST/Statements.h"

#include "Parser/Parser.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace llvm;

namespace rust_compiler::parser {

llvm::Expected<ast::Statements> Parser::parseStatements() {
  Location loc = getLocation();

  Statements stmts = {loc};

  while (true) {
    if (check(TokenKind::Eof)) {
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse statements: eof");
    } else if (check(TokenKind::BraceClose)) {
      // done
      return stmts;
    } else if (checkStatement()) {
      llvm::Expected<std::shared_ptr<ast::Statement>> stmt = parseStatement();
      if (auto e = stmt.takeError()) {
        llvm::errs() << "failed to parse statement in statements"
                     << std::move(e) << "\n";
        exit(EXIT_FAILURE);
      }
      stmts.addStmt(*stmt);
    } else if (checkExpressionWithoutBlock()) {
      llvm::Expected<std::shared_ptr<ast::Expression>> woBlock =
          parseExpressionWithoutBlock();
      if (auto e = woBlock.takeError()) {
        llvm::errs() << "failed to parse expression without block in statements"
                     << std::move(e) << "\n";
        exit(EXIT_FAILURE);
      }
      stmts.setTrailing(*woBlock);
    } else {
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse statements");
    }
  }

  return stmts;
}

} // namespace rust_compiler::parser
