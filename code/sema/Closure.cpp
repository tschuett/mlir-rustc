#include "AST/ClosureExpression.h"
#include "Sema/Sema.h"

using namespace rust_compiler::sema;

void Sema::analyzeClosureExpression(ast::ClosureExpression *close) {
  analyzeExpression(close->getBody().get());
}
