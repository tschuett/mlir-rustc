#include "AST/Statements.h"

#include "AST/Types/PrimitiveTypes.h"

#include <memory>

namespace rust_compiler::ast {

size_t Statements::getTokens() {
  size_t count = 0;

  if (onlySemi)
    return 1;

  for (auto &stmt : stmts)
    count += stmt->getTokens();

  if (trailing)
    count += (*trailing).getTokens();

  return count;
}

std::shared_ptr<ast::types::Type> Statements::getType() {

  if (trailing) {
    return trailing->getType();
  }

  return std::static_pointer_cast<ast::types::Type>(
      std::make_shared<types::PrimitiveType>(getLocation(),
                                             types::PrimitiveTypeKind::Unit));
}

} // namespace rust_compiler::ast
