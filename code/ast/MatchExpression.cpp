#include "AST/MatchExpression.h"

namespace rust_compiler::ast {

void MatchExpression::setScrutinee(Scrutinee scrut) { scrutinee = scrut; }

} // namespace rust_compiler::ast
