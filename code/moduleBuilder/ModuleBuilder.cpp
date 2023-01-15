#include "ModuleBuilder/ModuleBuilder.h"

#include "AST/Module.h"
#include "AST/Statement.h"
#include "AST/VariableDeclaration.h"
#include "Mir/MirDialect.h"
// #include "mlir/IR/AsmState.h"
#include "Mir/MirOps.h"

#include <llvm/Remarks/Remark.h>
#include <llvm/Target/TargetMachine.h>
#include <memory>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Verifier.h>
#include <optional>

namespace rust_compiler {

using namespace llvm;
using namespace mlir;

void ModuleBuilder::build(std::shared_ptr<ast::Module> mod) {
  llvm::outs() << "ModuleBuilder::build: " << mod->getItems().size() << "\n";

  for (auto i : mod->getItems()) {
    emitItem(i);
  }
  //  module->print(llvm::outs());
}

void ModuleBuilder::emitVisItem(std::shared_ptr<ast::VisItem> item) {
  llvm::outs() << "emitItem"
               << "\n";
  switch (item->getKind()) {
  case ast::VisItemKind::Function: {
    llvm::outs() << "found function"
                 << "\n";
    emitFun(std::static_pointer_cast<ast::Function>(item));
    break;
  }
  case ast::VisItemKind::Module: {
    llvm::outs() << "found module"
                 << "\n";
    emitModule(std::static_pointer_cast<ast::Module>(item));
    break;
  }
  case ast::VisItemKind::UseDeclaration: {
    llvm::outs() << "found use declaration"
                 << "\n";
    // buildFun(item);
    break;
  }
  default: {
    assert(false);
  }
  }
}

void ModuleBuilder::emitModule(std::shared_ptr<ast::Module> module) {
  // FIXME
}

} // namespace rust_compiler

// FIXME: rename loc and declare
