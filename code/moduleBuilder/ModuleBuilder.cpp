#include "ModuleBuilder/ModuleBuilder.h"

#include "AST/Module.h"
#include "AST/Statement.h"
#include "AST/VariableDeclaration.h"
#include "Mir/MirDialect.h"
// #include "mlir/IR/AsmState.h"
#include "Mir/MirOps.h"
#include "TypeBuilder.h"

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

void ModuleBuilder::build(std::shared_ptr<ast::Module> mod, Target &target) {
  llvm::outs() << "ModuleBuilder::build: " << mod->getItems().size() << "\n";

  for (auto i : mod->getItems()) {
    emitItem(i);
  }
  //  module->print(llvm::outs());
}

std::optional<mlir::Value>
ModuleBuilder::emitBlockExpression(std::shared_ptr<ast::BlockExpression> blk) {
  ScopedHashTableScope<llvm::StringRef,
                       std::pair<mlir::Value, ast::VariableDeclaration *>>
      varScope(symbolTable);

  std::optional<mlir::Value> result = std::nullopt;

  llvm::outs() << "emitBlockExpression"
               << "\n";

  // new variable scope?
  result = emitStatements(blk->getExpressions());

  return result;
}

// Declare a variable in the current scope, return success if the variable
// wasn't declared yet.
mlir::LogicalResult ModuleBuilder::declare(ast::VariableDeclaration &var,
                                           mlir::Value value) {
  if (symbolTable.count(var.getName()))
    return mlir::failure();
  symbolTable.insert(var.getName(), {value, &var});
  return mlir::success();
}

void ModuleBuilder::emitItem(std::shared_ptr<ast::Item> item) {
  llvm::outs() << "emitItem"
               << "\n";
  switch (item->getKind()) {
  case ast::ItemKind::Function: {
    llvm::outs() << "found function"
                 << "\n";
    emitFun(std::static_pointer_cast<ast::Function>(item));
    break;
  }
  case ast::ItemKind::Module: {
    llvm::outs() << "found module"
                 << "\n";
    emitModule(std::static_pointer_cast<ast::Module>(item));
    break;
  }
  case ast::ItemKind::InnerAttribute: {
    llvm::outs() << "found inner attribute"
                 << "\n";
    // buildFun(item);
    break;
  }
  case ast::ItemKind::UseDeclaration: {
    llvm::outs() << "found use declaration"
                 << "\n";
    // buildFun(item);
    break;
  }
  case ast::ItemKind::ClippyAttribute: {
    llvm::outs() << "found clippy attribute"
                 << "\n";
    // buildFun(item);
    break;
  }
  }
}

void ModuleBuilder::emitModule(std::shared_ptr<ast::Module> module) {
  // FIXME
}

mlir::Type ModuleBuilder::getType(std::shared_ptr<ast::types::Type> type) {
  return typeBuilder.getType(type);
}

} // namespace rust_compiler

// FIXME: rename loc and declare
