#include "ModuleBuilder/ModuleBuilder.h"

#include "AST/Statement.h"
#include "Mir/MirDialect.h"
// #include "mlir/IR/AsmState.h"
#include "Mir/MirOps.h"
#include "TypeBuilder.h"

#include <llvm/Remarks/Remark.h>
#include <llvm/Target/TargetMachine.h>
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
  for (auto f : mod->getFuncs()) {
    buildFun(f);
  }
}

std::optional<mlir::Value>
ModuleBuilder::emitBlockExpression(std::shared_ptr<ast::BlockExpression> blk) {
  ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);

  std::optional<mlir::Value> result = std::nullopt;

  // new variable scope?
  for (auto stmnt : blk->getExpressions()) {
    result = emitStatement(stmnt);
  }

  return result;
}

/// Declare a variable in the current scope, return success if the variable
/// wasn't declared yet.
// mlir::LogicalResult ModuleBuilder::declare(VarDeclExprAST &var,
//                                            mlir::Value value) {
//   if (symbolTable.count(var.getName()))
//     return mlir::failure();
//   symbolTable.insert(var.getName(), {value, &var});
//   return mlir::success();
// }

} // namespace rust_compiler

// FIXME: rename loc and declare
