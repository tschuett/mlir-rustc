#include "AST/LetStatement.h"
#include "Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::Statement>> Parser::parseLetStatement() {
  Location loc = getLocation();

  LetStatement let = {loc};

  if (checkOuterAttribute()) {
    llvm::Expected<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (auto e = outer.takeError()) {
      llvm::errs() << "failed to parse outer attributes: " << std::move(e)
                                                          << "\n";
      exit(EXIT_FAILURE);
    }
    let.setOuterAttributes(*outer);
  }

  if (!checkKeyWord(KeyWordKind::KW_LET)) {
    llvm::errs() << "failed to let token: "
                       << "\n";
    exit(EXIT_FAILURE);
  }

  assert(eatKeyWord(KeyWordKind::KW_LET));

  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>> pattern =
      parsePatternNoTopAlt();
  if (auto e = pattern.takeError()) {
    llvm::errs() << "failed to parse pattern no top alt: " << std::move(e)
                                                          << "\n";
    exit(EXIT_FAILURE);
  }
  let.setPattern(*pattern);

  if (check(TokenKind::Colon)) {
    assert(eat(TokenKind::Colon));
    llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> type =
        parseType();
    if (auto e = type.takeError()) {
      llvm::errs() << "failed to parse type: " << std::move(e) << "\n";
      exit(EXIT_FAILURE);
    }
    let.setType(*type);
  }

  if (check(TokenKind::Eq)) {
    assert(eat(TokenKind::Eq));
    llvm::Expected<std::shared_ptr<ast::Expression>> expr = parseExpression();
    if (auto e = expr.takeError()) {
      llvm::errs() << "failed to parse expression: " << std::move(e) << "\n";
      exit(EXIT_FAILURE);
    }
    let.setExpression(*expr);
  }

  if (checkKeyWord(KeyWordKind::KW_ELSE)) {
    assert(eatKeyWord(KeyWordKind::KW_ELSE));
    llvm::Expected<std::shared_ptr<ast::Expression>> block =
        parseBlockExpression();
    if (auto e = block.takeError()) {
      llvm::errs() << "failed to parse block expression: " << std::move(e) << "\n";
      exit(EXIT_FAILURE);
    }
    let.setElseExpr(*block);
  }

  if (!check(TokenKind::Semi)) {
    llvm::errs() << "failed to parse ; token:" <<  "\n";
      exit(EXIT_FAILURE);
  }

  assert(eat(TokenKind::Semi));

  return std::make_shared<LetStatement>(let);
}

} // namespace rust_compiler::parser
