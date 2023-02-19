#include "AST/FunctionParam.h"

#include "AST/Patterns/IdentifierPattern.h"
#include "AST/Types/TypeExpression.h"

#include <cassert>

namespace rust_compiler::ast {

void FunctionParam::setType(std::shared_ptr<ast::types::TypeExpression> _type) {
  type = _type;
}


} // namespace rust_compiler::ast
