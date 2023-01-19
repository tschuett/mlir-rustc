#include "ModuleBuilder/ModuleBuilder.h"

namespace rust_compiler {

// Declare a variable in the current scope, return success if the variable
// wasn't declared yet.
mlir::LogicalResult ModuleBuilder::declare(ast::VariableDeclaration &var,
                                           mlir::Value value) {
  llvm::outs() << "add variable to symbol table: " << var.getName() << "\n";
  if (symbolTable.contains(var.getName()))
    return mlir::failure();

  llvm::outs() << "add variable to symbol table (insert): x" << var.getName()
               << "x"
               << "\n";
  symbolTable.insert(var.getName(), {value, &var});
  llvm::outs() << "count: " << symbolTable.contains("right") << "\n";
  llvm::outs() << "count2: " << symbolTable.contains(var.getName()) << "\n";

  return mlir::success();
}

} // namespace rust_compiler
