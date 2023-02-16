#include "AST/ReturnExpression.h"

#include "AST/Types/Types.h"

namespace rust_compiler::ast {

std::shared_ptr<ast::Expression> ReturnExpression::getExpression() {
  return expr;
}

void ReturnExpression::setTail(std::shared_ptr<ast::Expression> _tail) {
  expr = _tail;
}

// std::shared_ptr<ast::types::Type> ReturnExpression::getType() {
//   if (expr)
//     return expr->getType();
//
//   return std::static_pointer_cast<ast::types::Type>(
//       std::make_shared<ast::types::PrimitiveType>(
//           getLocation(), types::PrimitiveTypeKind::Unit));
// }

} // namespace rust_compiler::ast

// For scientific articles surprise is great. You managed to do that? Wow. For
// performance, it is opposite. You don't want to surprise MPI. MPI: You want to
// me do what? This is the story of persistent collectives. An MPI_Start does
// not surprise MPI anymore. MPI: I knew that was coming. Maybe you can achieve
// the same for RMA. Register Ops at window generation. Then you can spend the
// rest of the day with MPI_RMA_Start. MPI: I knew that this was coming.
