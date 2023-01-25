#include "AST/ComparisonExpression.h"

#include "ModuleBuilder/ModuleBuilder.h"

#include <mlir/Dialect/Arith/IR/Arith.h>

using namespace rust_compiler::ast;
using namespace mlir;

namespace rust_compiler {

mlir::Value ModuleBuilder::emitComparisonExpression(
    std::shared_ptr<ast::ComparisonExpression> compare) {
  mlir::Value rhs = emitExpression(compare->getRHS());
  mlir::Value lhs = emitExpression(compare->getLHS());

  switch (compare->getKind()) {
  case ast::ComparisonExpressionKind::Equal: {
    return builder.create<arith::CmpIOp>(getLocation(compare->getLocation()),
                                         arith::CmpIPredicate::eq, lhs, rhs);
    break;
  }
  case ast::ComparisonExpressionKind::NotEqual: {
    return builder.create<arith::CmpIOp>(getLocation(compare->getLocation()),
                                         arith::CmpIPredicate::ne, lhs, rhs);
    break;
  }
  case ast::ComparisonExpressionKind::GreaterThan: {
    return builder.create<arith::CmpIOp>(getLocation(compare->getLocation()),
                                         arith::CmpIPredicate::ugt, lhs, rhs);
    break;
  }
  case ast::ComparisonExpressionKind::LessThan: {
    return builder.create<arith::CmpIOp>(getLocation(compare->getLocation()),
                                         arith::CmpIPredicate::slt, lhs, rhs);
    break;
  }
  case ast::ComparisonExpressionKind::GreaterThanOrEqualTo: {
    return builder.create<arith::CmpIOp>(getLocation(compare->getLocation()),
                                         arith::CmpIPredicate::uge, lhs, rhs);
    break;
  }
  case ast::ComparisonExpressionKind::LessThanOrEqualTo: {
    return builder.create<arith::CmpIOp>(getLocation(compare->getLocation()),
                                         arith::CmpIPredicate::ule, lhs, rhs);
    break;
  }
  }

  assert(false);
}

} // namespace rust_compiler


// FIXME: types and signs
