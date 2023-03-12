#include "AST/Statements.h"

#include "Lexer/Token.h"
#include "Parser/Parser.h"
#include "Parser/Restrictions.h"

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace llvm;

namespace rust_compiler::parser {

StringResult<ast::Statements> Parser::parseStatements() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();

  Statements stmts = {loc};

  llvm::errs() << "parseStatements"
               << "\n";

  Restrictions restrictions;

  while (true) {
    llvm::errs() << "parseStatements: " << Token2String(getToken().getKind())
                 << "\n";
    if (check(TokenKind::Eof)) {
      return StringResult<ast::Statements>("failed to parse statements: eof");
    } else if (check(TokenKind::Semi)) {
      // eat
      assert(eat(TokenKind::Semi));
    } else if (check(TokenKind::BraceClose)) {
      // done
      return StringResult<ast::Statements>(stmts);
    } else if (checkStatement()) {
      llvm::errs() << "parseStatements: statement"
                   << "\n";
      StringResult<std::shared_ptr<ast::Statement>> stmt =
          parseStatement(restrictions);
      if (!stmt) {
        std::string s =
            llvm::formatv(
                "{0} {1}",
                "failed to parse statement in statements: ", stmt.getError())
                .str();
        llvm::errs() << "failed to parse statement in statements: "
                     << stmt.getError() << "\n";
        printFunctionStack();
        //exit(EXIT_FAILURE);
        return StringResult<ast::Statements>(s);
      }
      stmts.addStmt(stmt.getValue());
    } else if (checkExpressionWithoutBlock()) {
      llvm::errs() << "parseStatements: wo block"
                   << "\n";
      StringResult<std::shared_ptr<ast::Expression>> woBlock =
          parseExpressionWithoutBlock({}, restrictions);
      if (!woBlock) {
        llvm::errs()
            << "failed to parse expression without block in statements: "
            << woBlock.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      stmts.setTrailing(woBlock.getValue());
    } else {
      llvm::errs() << "failed to parse statements"
                   << "\n";
      return StringResult<ast::Statements>("failed to parse statements");
    }
  }

  return StringResult<ast::Statements>(stmts);
}

} // namespace rust_compiler::parser
