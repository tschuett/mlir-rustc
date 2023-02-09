#pragma once

#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/InfiniteLoopExpression.h"
#include "AST/LiteralExpression.h"
#include "Sema/Common.h"

#include <memory>

namespace rust_compiler::ast {
class Expression;
class Crate;
} // namespace rust_compiler::ast

namespace rust_compiler::sema {
class Sema;
}

namespace rust_compiler::sema {

class TypeChecking {
  Sema *sema;

public:
  TypeChecking(Sema *sema) : sema(sema) {}

  //void eqExpr(AstId, AstId);
  void eqType(AstId, AstId);
  void isIntegerLike(AstId);
  void isFloatLike(AstId);
  void isIntegerOrFloatLike(AstId);
  void isKnownType(AstId, std::shared_ptr<ast::types::Type>);
  //void sub();

private:
//  void checkInfiniteLoopExpression(
//      std::shared_ptr<ast::InfiniteLoopExpression> loop);
//  void checkArithmeticOrLogicalExpression(
//      std::shared_ptr<ast::ArithmeticOrLogicalExpression> arith);
//  void checkLiteralExpression(std::shared_ptr<ast::LiteralExpression> lit);
};

} // namespace rust_compiler::sema
