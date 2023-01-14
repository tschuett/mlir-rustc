#include "Mir/MirOps.h"
#include "ModuleBuilder/ModuleBuilder.h"
#include "Remarks/OptimizationRemarkEmitter.h"

#include <llvm/Remarks/Remark.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/TypeRange.h>

using namespace llvm;
using namespace mlir;

using namespace rust_compiler::remarks;

namespace rust_compiler {

// static remarks::Remark createRemark(llvm::StringRef pass,
//                                     llvm::StringRef FunctionName) {
//   llvm::remarks::Remark r;
//   r.PassName = pass;
//   r.FunctionName = FunctionName;
//   return r;
// }

mlir::func::FuncOp ModuleBuilder::emitFun(std::shared_ptr<ast::Function> f) {
  ScopedHashTableScope<llvm::StringRef,
                       std::pair<mlir::Value, ast::VariableDeclaration *>>
      varScope(symbolTable);

  OptimizationRemarkEmitter ORE = {f, &serializer};

  llvm::outs() << "build function signature"
               << "\n";

  // Create an MLIR function for the given prototype.
  builder.setInsertionPointToEnd(theModule.getBody());

  mlir::func::FuncOp function =
      buildFunctionSignature(f->getSignature(), getLocation(f->getLocation()));
  if (!function)
    return nullptr;

  llvm::outs() << "declare function arguments: "
               << function.getArgumentTypes().size() << "\n";

  llvm::outs() << "declare function arguments: "
               << function.getBody().getNumArguments() << "\n";

  // Let's start the body of the function now!
  // mlir::Block &entryBlock = function.front();
  mlir::Block *entryBlock = currentBlock;
  auto protoArgs = f->getSignature().getParameters().getParams();

  llvm::outs() << "declare function arguments: "
               << entryBlock->getNumArguments() << "\n";

  llvm::outs() << "declare function arguments: " << entryBlock->isEntryBlock()
               << "\n";

  // assert(entryBlock.getNumArguments() == 1);

  // Declare all the function arguments in the symbol table.
  for (unsigned i = 0; i < entryBlock->getNumArguments(); ++i) {
    if (failed(declare(protoArgs[i], entryBlock->getArgument(i))))
      return nullptr;
  }

  llvm::outs() << "declared function arguments"
               << "\n";

  //  for (const auto nameValue : llvm::zip(protoArgs,
  //  entryBlock.getArguments())) {
  //    if (failed(declare(std::get<0>(nameValue), std::get<1>(nameValue))))
  //      return nullptr;
  //  }

  // Set the insertion point in the builder to the beginning of the function
  // body, it will be used throughout the codegen to create operations in this
  // function.
  builder.setInsertionPointToStart(entryBlock);

  // Emit the body of the function.
  emitBlockExpression(f->getBody());
  // FIXME returns optional

  // Implicitly return void if no return statement was emitted.
  // FIXME: we may fix the parser instead to always return the last expression
  // (this would possibly help the REPL case later)
  mlir::func::ReturnOp returnOp;
  if (!entryBlock->empty())
    returnOp = dyn_cast<mlir::func::ReturnOp>(entryBlock->back());
  if (!returnOp) {
    builder.create<mlir::func::ReturnOp>(getLocation(f->getLocation()));
  } else if (returnOp.getOperands().size() > 0) {
    // Otherwise, if this return operation has an operand then add a result to
    // the function.
    function.setType(
        builder.getFunctionType(function.getFunctionType().getInputs(),
                                *returnOp.operand_type_begin()));
  }

  llvm::outs() << "declared function HAPPY!"
               << "\n";

  //  // If this function isn't main, then set the visibility to private.
  //  if (funcAST.getProto()->getName() != "main")
  //    function.setPrivate();

  return function;
}

mlir::func::FuncOp
ModuleBuilder::buildFunctionSignature(ast::FunctionSignature sig,
                                      mlir::Location location) {
  SmallVector<mlir::Type, 10> argTypes;
  SmallVector<mlir::Location, 10> argLocations;
  ast::FunctionParameters params = sig.getParameters();

  llvm::outs() << "buildFunctionSignature"
               << "\n";

  for (auto &param : params.getParams()) {
    mlir::Type type = getType(param.getType());
    if (!type)
      return nullptr;

    argTypes.push_back(type);
    argLocations.push_back(getLocation(param.getLocation()));
  }

  mlir::Type resultType = getType(sig.getReturnType());

  llvm::outs() << "buildFunctionSignature: " << argTypes.size() << "\n";

  mlir::FunctionType funcType =
      builder.getFunctionType(argTypes, resultType /*std::nullopt results*/);

  mlir::func::FuncOp f =
      builder.create<mlir::func::FuncOp>(location, sig.getName(), funcType);
  assert(f.getArgumentTypes().size() == 1);
  assert(f.getResultTypes().size() == 1);

  llvm::outs() << "buildFunctionSignature: #blocks "
               << f.getBody().getBlocks().size() << "\n";

  currentBlock = builder.createBlock(
      &f.getBody(), {}, TypeRange(f.getArgumentTypes()), argLocations);

  llvm::outs() << "buildFunctionSignature: "
               << f.getBody().getArgumentTypes().size() << "\n";

  return f;
}

} // namespace rust_compiler
