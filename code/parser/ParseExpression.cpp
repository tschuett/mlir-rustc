#include "Parser/Parser.h"

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseExpression(Restrictions restrictions,
                        std::span<ast::OuterAttribute> outer) {
  return parseExpression(Precedence::Lowest, restrictions);
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseExpression(Precedence rightBindingPower,
                        std::span<ast::OuterAttribute> outer,
                        Restrictions restrictions) {
  CheckPoint cp = getCheckPoint();

  llvm::outs() << "parseExpression"
               << "\n";

  llvm::Expected<std::shared_ptr<ast::Expression>> left =
      parseUnaryExpression({}, restrictions);
  if (auto e = failed.takeError()) {
    llvm::errs() << "failed to parse unary expression in expression: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  

  // stop parsing if find lower priority token - parse higher priority first
  while (rightBindingPower < getLeftBindingPower(lexer.peek_token())) {
  }

  //  if (checkOuterAttribute()) {
  //    llvm::Expected<std::vector<ast::OuterAttribute>> outer =
  //        parseOuterAttributes();
  //    if (auto e = outer.takeError()) {
  //      llvm::errs() << "failed to parse outer attributes in expression: "
  //                   << toString(std::move(e)) << "\n";
  //      exit(EXIT_FAILURE);
  //    }
  //  }
  //
  //  if (checkKeyWord(KeyWordKind::KW_LOOP)) {
  //    recover(cp);
  //    return parseExpressionWithBlock();
  //  } else if (checkKeyWord(KeyWordKind::KW_MATCH)) {
  //    recover(cp);
  //    return parseExpressionWithBlock();
  //  } else if (checkKeyWord(KeyWordKind::KW_WHILE)) {
  //    recover(cp);
  //    return parseExpressionWithBlock();
  //  } else if (checkKeyWord(KeyWordKind::KW_IF)) {
  //    recover(cp);
  //    return parseExpressionWithBlock();
  //  } else if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
  //    recover(cp);
  //    return parseExpressionWithBlock();
  //  } else if (checkKeyWord(KeyWordKind::KW_FOR)) {
  //    recover(cp);
  //    return parseExpressionWithBlock();
  //  } else if (check(TokenKind::BraceOpen)) {
  //    recover(cp);
  //    return parseExpressionWithBlock();
  //  } else if (check(TokenKind::LIFETIME_OR_LABEL) && check(TokenKind::Colon))
  //  {
  //    recover(cp);
  //    return parseExpressionWithBlock();
  //  }
  //
  //  recover(cp);
  //  return parseExpressionWithoutBlock();
}

} // namespace rust_compiler::parser
