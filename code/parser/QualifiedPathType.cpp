#include "AST/Types/QualifiedPathType.h"

namespace rust_compiler::ast::types {

void QualifiedPathType::setType(
    std::shared_ptr<ast::types::TypeExpression> _type) {
  type = _type;
}

void QualifiedPathType::setPath(std::shared_ptr<ast::types::TypePath> _path) {
  path = _path;
}

} // namespace rust_compiler::ast::types
