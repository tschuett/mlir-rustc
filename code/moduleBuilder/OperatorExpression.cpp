#include "AST/ArithmeticOrLogicalExpression.h"
#include "ModuleBuilder/ModuleBuilder.h"

namespace rust_compiler {

mlir::Value ModuleBuilder::emitArithmeticOrLogicalExpression(
    std::shared_ptr<ast::ArithmeticOrLogicalExpression> expr) {
  mlir::Value rhs = emitExpression(expr->getRHS());

  switch (expr->getKind()) {}
}

} // namespace rust_compiler
