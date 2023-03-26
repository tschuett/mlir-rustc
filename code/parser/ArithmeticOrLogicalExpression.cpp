#include "AST/ArithmeticOrLogicalExpression.h"

#include "Parser/Parser.h"

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace rust_compiler::lexer;
using namespace llvm;

namespace rust_compiler::parser {

StringResult<std::shared_ptr<ast::Expression>>
Parser::parseArithmeticOrLogicalExpression(std::shared_ptr<ast::Expression> lhs,
                                           Restrictions restrictions) {
  Location loc = getLocation();

  ArithmeticOrLogicalExpression arith = {loc};
  arith.setLhs(lhs);

  //  llvm::outs() << "parseArithmeticOrLogicalExpression"
  //               << "\n";

  if (check(TokenKind::Plus)) {
    arith.setKind(ArithmeticOrLogicalExpressionKind::Addition);
  } else if (check(TokenKind::Minus)) {
    arith.setKind(ArithmeticOrLogicalExpressionKind::Subtraction);
  } else if (check(TokenKind::Star)) {
    arith.setKind(ArithmeticOrLogicalExpressionKind::Multiplication);
  } else if (check(TokenKind::Slash)) {
    arith.setKind(ArithmeticOrLogicalExpressionKind::Division);
  } else if (check(TokenKind::Percent)) {
    arith.setKind(ArithmeticOrLogicalExpressionKind::Remainder);
  } else if (check(TokenKind::And)) {
    arith.setKind(ArithmeticOrLogicalExpressionKind::BitwiseAnd);
  } else if (check(TokenKind::Or)) {
    arith.setKind(ArithmeticOrLogicalExpressionKind::BitwiseOr);
  } else if (check(TokenKind::Caret)) {
    arith.setKind(ArithmeticOrLogicalExpressionKind::BitwiseXor);
  } else if (check(TokenKind::Shl)) {
    arith.setKind(ArithmeticOrLogicalExpressionKind::LeftShift);
  } else if (check(TokenKind::Shr)) {
    arith.setKind(ArithmeticOrLogicalExpressionKind::RightShift);
  } else {
    return StringResult<std::shared_ptr<ast::Expression>>(
        "failed to parse arithmetic or logical expression");
  }

  // the operator
  assert(eat(getToken().getKind()));

  StringResult<std::shared_ptr<ast::Expression>> expr =
      parseExpression({}, restrictions);
  if (!expr) {
    llvm::errs()
        << "failed to parse expression in arithmetic or logical expression: "
        << expr.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  arith.setRhs(expr.getValue());

  return StringResult<std::shared_ptr<ast::Expression>>(
      std::make_shared<ArithmeticOrLogicalExpression>(arith));
}

} // namespace rust_compiler::parser
