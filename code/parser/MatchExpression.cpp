
#include "AST/MatchExpression.h"

#include "AST/Expression.h"
#include "AST/Scrutinee.h"
#include "Lexer/KeyWords.h"
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

StringResult<ast::MatchArmGuard> Parser::parseMatchArmGuard() {
  Location loc = getLocation();
  MatchArmGuard guard = {loc};

  if (!checkKeyWord(KeyWordKind::KW_IF))
    return StringResult<ast::MatchArmGuard>("failed to parse match arm guard");

  assert(checkKeyWord(KeyWordKind::KW_IF));

  Restrictions restrictions;
  StringResult<std::shared_ptr<ast::Expression>> expr =
      parseExpression({}, restrictions);
  if (!expr) {
    llvm::errs() << "failed to parse expression in match arm guard: "
                 << expr.getError() << "\n";
    printFunctionStack();
    std::string s =
        llvm::formatv(
            "{0} {1}",
            "failed to parse expression in match arm guard: ", expr.getError())
            .str();
    return StringResult<ast::MatchArmGuard>(s);
  }

  guard.setGuard(expr.getValue());

  return StringResult<ast::MatchArmGuard>(guard);
}

StringResult<ast::MatchArm> Parser::parseMatchArm() {
  Location loc = getLocation();
  MatchArm arm = {loc};

  if (checkOuterAttribute()) {
    StringResult<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (!outer) {
      llvm::errs() << "failed to parse outer attributes in match arm: "
                   << outer.getError() << "\n";
      printFunctionStack();
      // exit(EXIT_FAILURE);
      std::string s =
          llvm::formatv("{0} {1}",
                        "failed to parse outer attributes in match arms: ",
                        outer.getError())
              .str();
      return StringResult<ast::MatchArm>(s);
    }
    std::vector<ast::OuterAttribute> ot = outer.getValue();
    arm.setOuterAttributes(ot);
  }

  StringResult<std::shared_ptr<ast::patterns::Pattern>> pattern =
      parsePattern();
  if (!pattern) {
    llvm::errs() << "failed to parse pattern in match arm: "
                 << pattern.getError() << "\n";
    printFunctionStack();
    std::string s =
        llvm::formatv("{0} {1}", "failed to parse pattern in match arm: ",
                      pattern.getError())
            .str();
    return StringResult<ast::MatchArm>(s);
  }
  arm.setPattern(pattern.getValue());

  if (checkKeyWord(KeyWordKind::KW_IF)) {
    StringResult<ast::MatchArmGuard> guard = parseMatchArmGuard();
    if (!guard) {
      llvm::errs() << "failed to parse match arm guard in match arm: "
                   << guard.getError() << "\n";
      printFunctionStack();
      std::string s =
          llvm::formatv("{0} {1}",
                        "failed to parse match arm guard in match arm: ",
                        guard.getError())
              .str();
      return StringResult<ast::MatchArm>(s);
    }
    arm.setGuard(guard.getValue());
  }

  return StringResult<ast::MatchArm>(arm);
}

StringResult<ast::MatchArms> Parser::parseMatchArms() {
  Location loc = getLocation();
  MatchArms arms = {loc};

  StringResult<ast::MatchArm> arm = parseMatchArm();
  if (!arm) {
    llvm::errs() << "failed to parse match arm in match arms: "
                 << arm.getError() << "\n";
    printFunctionStack();
    std::string s =
        llvm::formatv("{0} {1}", "failed to parse match arm in match arms: ",
                      arm.getError())
            .str();
    return StringResult<ast::MatchArms>(s);
  }

  if (!check(TokenKind::FatArrow)) {
    llvm::errs() << Token2String(getToken().getKind()) << "\n";
    return StringResult<ast::MatchArms>(
        "failed to parse fat arrow in match arms");
  }
  assert(eat(TokenKind::FatArrow));

  Restrictions restrictions;
  StringResult<std::shared_ptr<Expression>> expr =
      parseExpression({}, restrictions);
  if (!expr) {
    llvm::errs() << "failed to parse expression in match arms: "
                 << expr.getError() << "\n";
    printFunctionStack();
    std::string s =
        llvm::formatv("{0} {1}", "failed to parse expression in match arms: ",
                      expr.getError())
            .str();
    return StringResult<ast::MatchArms>(s);
  }

  arms.addArm(arm.getValue(), expr.getValue());

  if ((expr.getValue())->getExpressionKind() ==
      ExpressionKind::ExpressionWithoutBlock) {
    if (!check(TokenKind::Comma)) {
      return StringResult<ast::MatchArms>(
          "failed to parse , after parse expression without "
          "block in match arms");
    }
    assert(eat(TokenKind::Comma));
  } else {
    if (check(TokenKind::Comma))
      assert(eat(TokenKind::Comma));
  }

  while (true) {
    if (check(TokenKind::Eof)) {
      // abort
      return StringResult<ast::MatchArms>("failed to parse match arms: eof");
    } else if (check(TokenKind::BraceClose)) {
      // done
      return StringResult<ast::MatchArms>(arms);
    } else if (check(TokenKind::Comma) && check(TokenKind::BraceClose, 1)) {
      // done
      assert(eat(TokenKind::Comma));
      return StringResult<ast::MatchArms>(arms);
    } else if (!check(TokenKind::BraceClose)) {
      StringResult<ast::MatchArm> arm = parseMatchArm();
      if (!arm) {
        llvm::errs() << "failed to parse match arm in match arms: "
                     << arm.getError() << "\n";
        printFunctionStack();
        std::string s =
            llvm::formatv(
                "{0} {1}",
                "failed to parse match arm in match arms: ", arm.getError())
                .str();
        return StringResult<ast::MatchArms>(s);
      }
      if (!check(TokenKind::FatArrow)) {
        return StringResult<ast::MatchArms>(
            "failed to parse fat arrow in match arms");
      }
      assert(eat(TokenKind::FatArrow));

      Restrictions restrictions;
      StringResult<std::shared_ptr<Expression>> expr =
          parseExpression({}, restrictions);
      if (!expr) {
        llvm::errs() << "failed to parse expression in match arms: "
                     << expr.getError() << "\n";
        printFunctionStack();
        std::string s =
            llvm::formatv(
                "{0} {1}",
                "failed to parse expression in match arms: ", expr.getError())
                .str();
        return StringResult<ast::MatchArms>(s);
      }
      arms.addArm(arm.getValue(), expr.getValue());

      if ((expr.getValue())->getExpressionKind() ==
          ExpressionKind::ExpressionWithoutBlock) {
        if (!check(TokenKind::Comma)) {
          return StringResult<ast::MatchArms>(
              "failed to parse , after parse expression without "
              "block in match arms");
        }
        assert(eat(TokenKind::Comma));
      } else {
        if (check(TokenKind::Comma))
          assert(eat(TokenKind::Comma));
      }
    } else {
      // error
      llvm::errs() << "parseMatchArms unknown token: "
                   << Token2String(getToken().getKind()) << "\n";
      return StringResult<ast::MatchArms>(
          "failed to parse match arms with unknown token");
    }
  }
  return StringResult<ast::MatchArms>("failed to parse match arms");
}

StringResult<std::shared_ptr<ast::Expression>>
Parser::parseMatchExpression(std::span<ast::OuterAttribute>) {
  Location loc = getLocation();
  MatchExpression ma = {loc};

  if (checkKeyWord(KeyWordKind::KW_MATCH)) {
    assert(eatKeyWord(KeyWordKind::KW_MATCH));
  } else {
    return StringResult<std::shared_ptr<ast::Expression>>(
        "failed to parse match keyword in match expression");
  }

  StringResult<ast::Scrutinee> scrutinee = parseScrutinee();
  if (!scrutinee) {
    llvm::errs() << "failed to parse scrutinee in match expression: "
                 << scrutinee.getError() << "\n";
    printFunctionStack();
    std::string s =
        llvm::formatv("{0} {1}",
                      "failed to parse scrutinee in match expression: ",
                      scrutinee.getError())
            .str();
    return StringResult<std::shared_ptr<ast::Expression>>(s);
  }
  ma.setScrutinee(scrutinee.getValue());

  if (check(TokenKind::BraceOpen)) {
    assert(eat(TokenKind::BraceOpen));
  } else {
    return StringResult<std::shared_ptr<ast::Expression>>(
        "failed to parse { token in match expression");
  }

  if (checkInnerAttribute()) {
    StringResult<std::vector<ast::InnerAttribute>> inner =
        parseInnerAttributes();
    if (!inner) {
      llvm::errs() << "failed to parse inner attribute in match expression: "
                   << inner.getError() << "\n";
      printFunctionStack();
      std::string s =
          llvm::formatv("{0} {1}",
                        "failed to parse inner attribute in match expression: ",
                        inner.getError())
              .str();
      return StringResult<std::shared_ptr<ast::Expression>>(s);
    }
    std::vector<ast::InnerAttribute> in = inner.getValue();
    ma.setInnerAttributes(in);
  }

  if (check(TokenKind::BraceClose)) {
    assert(eat(TokenKind::BraceClose));
    return StringResult<std::shared_ptr<ast::Expression>>(
        std::make_shared<MatchExpression>(ma));
  }

  StringResult<ast::MatchArms> arms = parseMatchArms();
  if (!arms) {
    llvm::errs() << "failed to parse match arms in match expression: "
                 << arms.getError() << "\n";
    printFunctionStack();
    std::string s =
        llvm::formatv(
            "{0} {1}",
            "failed to parse match arms in match expression: ", arms.getError())
            .str();
    return StringResult<std::shared_ptr<ast::Expression>>(s);
  }
  ma.setMatchArms(arms.getValue());

  if (check(TokenKind::BraceClose)) {
    assert(eat(TokenKind::BraceClose));
    return StringResult<std::shared_ptr<ast::Expression>>(
        std::make_shared<MatchExpression>(ma));
  }

  llvm::errs() << "parseMatchExpression: " << Token2String(getToken().getKind())
               << "\n";
  return StringResult<std::shared_ptr<ast::Expression>>(
      "failed to parse match expression");
}

} // namespace rust_compiler::parser
