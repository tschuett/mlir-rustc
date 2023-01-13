#include "Sema/Sema.h"

#include "AST/Module.h"

namespace rust_compiler::sema {

void analyzeSemantics(std::shared_ptr<ast::Module> &ast) {

  Sema sema;

  sema.analyze(ast);
}

} // namespace rust_compiler::sema
