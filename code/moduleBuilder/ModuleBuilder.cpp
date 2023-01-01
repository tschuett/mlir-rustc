#include "ModuleBuilder.h"

#include "AST/Statement.h"
#include "Mir/MirDialect.h"
// #include "mlir/IR/AsmState.h"
#include "Mir/MirOps.h"
#include "TypeBuilder.h"
#include "mlir/IR/SymbolTable.h"

#include <llvm/Remarks/Remark.h>
#include <llvm/Target/TargetMachine.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Verifier.h>

namespace rust_compiler {

using namespace llvm;
using namespace mlir;

remarks::Remark createRemark(llvm::StringRef pass,
                             llvm::StringRef FunctionName) {
  llvm::remarks::Remark r;
  r.PassName = pass;
  r.FunctionName = FunctionName;
  return r;
}

void ModuleBuilder::build(std::shared_ptr<ast::Module> mod, Target &target) {
  for (auto f : mod->getFuncs()) {
    buildFun(f);
  }
}


mlir::LogicalResult
ModuleBuilder::buildBlockExpression(std::shared_ptr<ast::BlockExpression> blk) {
  // new variable scope?
  for (auto stmnt : blk->getExpressions()) {
    buildStatement(stmnt);
  }

  return LogicalResult::success();
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
