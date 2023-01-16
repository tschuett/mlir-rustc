#include "AST/ReturnExpression.h"

#include "AST/Types/PrimitiveTypes.h"
#include "AST/Types/Types.h"

namespace rust_compiler::ast {

size_t ReturnExpression::getTokens() {
  size_t count = 1;

  if (expr)
    count += expr->getTokens();

  if (getHasTrailingSemi())
    ++count;

  return count;
}

std::shared_ptr<ast::Expression> ReturnExpression::getExpression() {
  return expr;
}

std::shared_ptr<ast::types::Type> ReturnExpression::getType() {
  if (expr)
    return expr->getType();

  return std::static_pointer_cast<ast::types::Type>(
      std::make_shared<ast::types::PrimitiveType>(
          getLocation(), types::PrimitiveTypeKind::Unit));
}

} // namespace rust_compiler::ast
