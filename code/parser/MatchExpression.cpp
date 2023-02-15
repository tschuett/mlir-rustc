#include "AST/MatchExpression.h"

#include "AST/Scrutinee.h"
#include "Lexer/KeyWords.h"
#include "Parser/Parser.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;

namespace rust_compiler::parser {

llvm::Expected<ast::MatchArmGuard> Parser::parseMatchGuard() {}

llvm::Expected<ast::MatchArm> Parser::parseMatchArm() {
  Location loc = getLocation();
  MatchArm arm = {loc};

  if (checkOuterAttribute()) {
    llvm::Expected<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    // check error
  }

  llvm::Expected<ast::patterns::Pattern> pattern = parsePattern();
  // check error

  if (checkKeyWord(KeyWordKind::KW_IF)) {
    llvm::Expected<ast::MatchArmGuard> guard = parseMatchGuard();
    // check guard
  }

  // fixme
}

llvm::Expected<ast::MatchArms> Parser::parseMatchArms() {
  Location loc = getLocation();
  MatchArms arms = {loc};

  llvm::Expected<ast::MatchArm> arm = parseMatchArm();
  // fixme
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseMatchExpression() {
  Location loc = getLocation();
  MatchExpression ma = {loc};

  if (checkKeyWord(KeyWordKind::KW_MATCH)) {
    assert(eatKeyWord(KeyWordKind::KW_MATCH));
  } else {
    // check error
  }

  llvm::Expected<ast::Scrutinee> scrutinee = parseScrutinee();
  if (auto e = scrutinee.takeError()) {
    llvm::errs() << "failed to parse scrutinee in match expression: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  ma.setScrutinee(*scrutinee);

  if (check(TokenKind::BraceOpen)) {
    assert(eat(TokenKind::BraceOpen));
  } else {
    // check error
  }

  if (checkInnerAttribute()) {
    llvm::Expected<std::vector<ast::InnerAttribute>> inner =
        parseInnerAttributes();
    // check error
  }

  // FIXME
}

} // namespace rust_compiler::parser
