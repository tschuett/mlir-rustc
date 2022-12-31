#include "ModuleBuilder.h"

#include "Mir/MirDialect.h"
// #include "mlir/IR/AsmState.h"
#include "TypeBuilder.h"

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

mlir::func::FuncOp ModuleBuilder::buildFun(std::shared_ptr<ast::Function> f) {
  ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);

  // serializer.emit(createRemark("codegen", f->getName()));
  serializer.emit(createRemark("codegen", "fun"));

  builder.setInsertionPointToEnd(theModule.getBody());
  mlir::func::FuncOp function =
      buildFunctionSignature(f->getSignature(), f->getLocation());
  if (!function)
    return nullptr;

  // Let's start the body of the function now!
  mlir::Block &entryBlock = function.front();
  auto protoArgs = f->getSignature().getArgs();

  // Declare all the function arguments in the symbol table.
  for (const auto nameValue : llvm::zip(protoArgs, entryBlock.getArguments())) {
    if (failed(declare(*std::get<0>(nameValue), std::get<1>(nameValue))))
      return nullptr;
  }

  // Set the insertion point in the builder to the beginning of the function
  // body, it will be used throughout the codegen to create operations in this
  // function.
  builder.setInsertionPointToStart(&entryBlock);

  // Emit the body of the function.
  if (mlir::failed(buildBlockExpression(f->getBody()))) {
    function.erase();
    return nullptr;
  }

  // Implicitly return void if no return statement was emitted.
  // FIXME: we may fix the parser instead to always return the last expression
  // (this would possibly help the REPL case later)
  Mir::ReturnOp returnOp;
  if (!entryBlock.empty())
    returnOp = dyn_cast<Mir::ReturnOp>(entryBlock.back());
  if (!returnOp) {
    builder.create<Mir::ReturnOp>(f->getSignature().getLocation());
  } else if (returnOp.operands().getType().size() == 0) { // FIXME: odd
    // Otherwise, if this return operation has an operand then add a result to
    // the function.
    function.setType(builder.getFunctionType(function.operands(),
                                             *returnOp.operand_type_begin()));
  }

  //  // If this function isn't main, then set the visibility to private.
  //  if (funcAST.getProto()->getName() != "main")
  //    function.setPrivate();

  return function;
}

mlir::func::FuncOp ModuleBuilder::buildFunctionSignature(ast::FunctionSignature sig,
                                                  mlir::Location location) {
  SmallVector<mlir::Type, 10> argTypes;
  TypeBuilder typeBuilder;

  for (auto &arg : sig.getArgs()) {
    mlir::Type type = typeBuilder.getType(arg.getType());
    if (!type)
      return nullptr;

    argTypes.push_back(type);
  }

  mlir::Type resultType = typeBuilder.getType(sig.getResult());

  auto funcType =
      builder.getFunctionType(argTypes, resultType /*std::nullopt results*/);
  return builder.create<mlir::func::FuncOp>(location, sig.getName(), funcType);
}

mlir::LogicalResult
ModuleBuilder::buildBlockExpression(std::shared_ptr<ast::BlockExpression> blk) {
  return LogicalResult::success();
}

/// Declare a variable in the current scope, return success if the variable
/// wasn't declared yet.
mlir::LogicalResult ModuleBuilder::declare(VarDeclExprAST &var,
                                           mlir::Value value) {
  if (symbolTable.count(var.getName()))
    return mlir::failure();
  symbolTable.insert(var.getName(), {value, &var});
  return mlir::success();
}

} // namespace rust_compiler

// FIXME: rename loc and declare
