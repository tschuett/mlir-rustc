#include "AST/Statements.h"

#include "Parser/Parser.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace llvm;

namespace rust_compiler::parser {

llvm::Expected<ast::Statements> Parser::parseStatements() {
  Location loc = getLocation();

  Statements stmts = {loc};

  xxx

      llvm::Expected<std::shared_ptr<ast::Expression>>
          woBlock = parseExpressionWithoutBlock();
  if (auto e = woBlock.takeError()) {
    llvm::errs() << "failed to parse expression without block" << std::move(e)
                 << "\n";
    exit(EXIT_FAILURE);
  }
  Statements stmts = {loc};
  stmts.setTrailing(woBlock);

  return std::make_shared<Statements>(stmts);
}

} // namespace rust_compiler::parser
