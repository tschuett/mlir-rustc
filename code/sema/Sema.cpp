#include "Sema/Sema.h"

#include "AST/Module.h"

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

void analyzeSemantics(std::shared_ptr<ast::Module> &ast) {

  Sema sema;

  sema.analyze(ast);
}

void Sema::analyze(std::shared_ptr<ast::Module> &ast) {
  // FIXME

  switch (ast->getModuleKind()) {
  case ModuleKind::Module: {
    break;
  }
  case ModuleKind::ModuleTree: {
    for (auto &item : ast->getItems()) {
      walkItem(item);
    }
    break;
  }
  case ModuleKind::Outer: {
    for (auto &item : ast->getItems()) {
      walkItem(item);
    }
    break;
  }
  }
}

} // namespace rust_compiler::sema
