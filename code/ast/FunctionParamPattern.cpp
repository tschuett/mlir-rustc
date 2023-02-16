#include "AST/FunctionParamPattern.h"

namespace rust_compiler::ast {

void FunctionParamPattern::setName(
    std::shared_ptr<ast::patterns::PatternNoTopAlt> _name) {
  name = _name;
}

void FunctionParamPattern::setType(
    std::shared_ptr<ast::types::TypeExpression> _type) {
  type = _type;
}

} // namespace rust_compiler::ast
