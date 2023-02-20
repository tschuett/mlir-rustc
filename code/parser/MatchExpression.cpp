#include "AST/MatchExpression.h"

#include "AST/Scrutinee.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace llvm;

namespace rust_compiler::parser {

llvm::Expected<ast::MatchArmGuard> Parser::parseMatchArmGuard() {
  Location loc = getLocation();
  MatchArmGuard guard = {loc};

  if (!checkKeyWord(KeyWordKind::KW_IF))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse match arm guard");

  assert(checkKeyWord(KeyWordKind::KW_IF));

  llvm::Expected<std::shared_ptr<ast::Expression>> expr = parseExpression();
  if (auto e = expr.takeError()) {
    llvm::errs() << "failed to parse expression in match arm guard: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  guard.setGuard(*expr);

  return guard;
}

llvm::Expected<ast::MatchArm> Parser::parseMatchArm() {
  Location loc = getLocation();
  MatchArm arm = {loc};

  if (checkOuterAttribute()) {
    llvm::Expected<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (auto e = outer.takeError()) {
      llvm::errs() << "failed to parse outer attributes in match guard: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    arm.setOuterAttributes(*outer);
  }

  llvm::Expected<std::shared_ptr<ast::patterns::Pattern>> pattern =
      parsePattern();
  if (auto e = pattern.takeError()) {
    llvm::errs() << "failed to parse pattern in match guard: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  arm.setPattern(*pattern);

  if (checkKeyWord(KeyWordKind::KW_IF)) {
    llvm::Expected<ast::MatchArmGuard> guard = parseMatchArmGuard();
    if (auto e = guard.takeError()) {
      llvm::errs() << "failed to parse match arm guard in match guard: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    arm.setGuard(*guard);
  }

  return arm;
}

llvm::Expected<ast::MatchArms> Parser::parseMatchArms() {
  Location loc = getLocation();
  MatchArms arms = {loc};

  llvm::Expected<ast::MatchArm> arm = parseMatchArm();
  if (auto e = arm.takeError()) {
    llvm::errs() << "failed to parse match arm in match arms: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  if (!check(TokenKind::FatArrow)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse fat arrow in match arms");
  }
  assert(eat(TokenKind::FatArrow));

  while (true) {
    if (check(TokenKind::Eof)) {
    } else if (check(TokenKind::BraceClose)) {
    } else if (checkOuterAttribute()) {
    }
  }

  xxx;
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseMatchExpression() {
  Location loc = getLocation();
  MatchExpression ma = {loc};

  if (checkKeyWord(KeyWordKind::KW_MATCH)) {
    assert(eatKeyWord(KeyWordKind::KW_MATCH));
  } else {
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse match keyword in match expression");
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
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse { token in match expression");
  }

  if (checkInnerAttribute()) {
    llvm::Expected<std::vector<ast::InnerAttribute>> inner =
        parseInnerAttributes();
    if (auto e = inner.takeError()) {
      llvm::errs() << "failed to parse inner attributes in match expression: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    ma.setInnerAttributes(*inner);
  }

  if (check(TokenKind::BraceClose)) {
    assert(eat(TokenKind::BraceClose));
    return std::make_shared<MatchExpression>(ma);
  }

  llvm::Expected<ast::MatchArms> arms = parseMatchArms();
  if (auto e = arms.takeError()) {
    llvm::errs() << "failed to parse match arms in match expression: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  ma.setMatchArms(*arms);

  if (check(TokenKind::BraceClose)) {
    assert(eat(TokenKind::BraceClose));
    return std::make_shared<MatchExpression>(ma);
  }

  return createStringError(inconvertibleErrorCode(),
                           "failed to parse match expression");
}

} // namespace rust_compiler::parser
