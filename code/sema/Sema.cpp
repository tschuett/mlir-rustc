#include "Sema/Sema.h"

#include "AST/Module.h"

namespace rust_compiler::sema {

void analyzeSemantics(std::shared_ptr<ast::Module> &ast) {

  Sema sema;

  sema.analyze(ast);
}

void Sema::analyze(std::shared_ptr<ast::Module> &ast) {
  // FIXME
  assert(false);
}

} // namespace rust_compiler::sema
