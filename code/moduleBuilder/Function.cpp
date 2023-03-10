#include "Mir/MirOps.h"
#include "ModuleBuilder/ModuleBuilder.h"
#include "Remarks/OptimizationRemarkEmitter.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Remarks/Remark.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/TypeRange.h>

using namespace llvm;
using namespace mlir;

using namespace rust_compiler::remarks;
using namespace rust_compiler::ast;

namespace rust_compiler {

// static remarks::Remark createRemark(llvm::StringRef pass,
//                                     llvm::StringRef FunctionName) {
//   llvm::remarks::Remark r;
//   r.PassName = pass;
//   r.FunctionName = FunctionName;
//   return r;
// }

mlir::func::FuncOp ModuleBuilder::emitFun(std::shared_ptr<ast::Function> f) {
  adt::ScopedHashTableScope<std::string,
                            std::pair<mlir::Value, ast::VariableDeclaration *>>
      _FunScope(symbolTable);

  OptimizationRemarkEmitter ORE = {f, &serializer};

  // Create an MLIR function for the given prototype.
  builder.setInsertionPointToEnd(theModule.getBody());

  mlir::func::FuncOp function =
      buildFunctionSignature(f->getSignature(), getLocation(f->getLocation()));
  if (!function)
    return nullptr;

  // Let's start the body of the function now!
  mlir::Block &entryBlock = function.front();
  std::vector<FunctionParam> protoArgs =
      f->getSignature().getParameters().getParams();

  // Declare all the function arguments in the symbol table.
  for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
    if (failed(declare(protoArgs[i], entryBlock.getArgument(i)))) {
      return nullptr;
    } else {
      llvm::outs() << "outside count: "
                   << symbolTable.contains(protoArgs[i].getName()) << "\n";
    }
  }

  // symbolTable.insert(protoArgs[0].getName(), {entryBlock->getArgument(0),
  // &protoArgs[0]});

  llvm::outs() << "declared function arguments"
               << "\n";

  llvm::outs() << "count: " << symbolTable.contains("right") << "\n";

  assert(symbolTable.contains("right"));

  //  for (const auto nameValue : llvm::zip(protoArgs,
  //  entryBlock.getArguments())) {
  //    if (failed(declare(std::get<0>(nameValue), std::get<1>(nameValue))))
  //      return nullptr;
  //  }

  // Set the insertion point in the builder to the beginning of the function
  // body, it will be used throughout the codegen to create operations in this
  // function.
  builder.setInsertionPointToStart(&entryBlock);

  // Emit the body of the function.
  emitBlockExpression(f->getBody());
  // FIXME returns optional

  // Implicitly return void if no return statement was emitted.
  // FIXME: we may fix the parser instead to always return the last expression
  // (this would possibly help the REPL case later)
  // mlir::func::ReturnOp returnOp;
  // if (!entryBlock->empty())
  //  returnOp = dyn_cast<mlir::func::ReturnOp>(entryBlock->back());
  // if (!returnOp) {
  //  builder.create<mlir::func::ReturnOp>(getLocation(f->getLocation()));
  //} else if (returnOp.getOperands().size() > 0) {
  //  // Otherwise, if this return operation has an operand then add a result to
  //  // the function.
  //  function.setType(
  //      builder.getFunctionType(function.getFunctionType().getInputs(),
  //                              *returnOp.operand_type_begin()));
  //}

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

  mlir::FunctionType funcType =
      builder.getFunctionType(argTypes, resultType /*std::nullopt results*/);

  llvm::SmallVector<mlir::NamedAttribute> attrs;
  //  attrs.push_back(
  //      builder.getNamedAttr("visibility", builder.getStringAttr("pub")));
  //  attrs.push_back(
  //      builder.getNamedAttr("function type",
  //      builder.getStringAttr("async")));

  mlir::func::FuncOp f = builder.create<mlir::func::FuncOp>(
      location, sig.getName(), funcType, attrs);
  //assert(f.getArgumentTypes().size() == 1);
  //assert(f.getResultTypes().size() == 1);

  entryBlock =
    builder.createBlock(&f.getBody(), f.getBody().end(),
                          TypeRange(f.getArgumentTypes()), argLocations);
  //
  //  entryBlock =
  //    builder.createBlock(f.getBody().front(),
  //                          TypeRange(f.getArgumentTypes()), argLocations);

  // entryBlock = builder.createBlock( // FIXME: BAD
  //     &f.getBody(), {}, TypeRange(f.getArgumentTypes()), argLocations);
  //
  //  region variant

  //  entryBlock = // FIXME: bad
  //    builder.createBlock(&f.getBody(), f.getBody().end(),
  //                          TypeRange(f.getArgumentTypes()), argLocations);
  //
  // block variant

  //  entryBlock = builder.createBlock(
  //      &f.getBody().back(), TypeRange(f.getArgumentTypes()), argLocations);
  //

  //    entryBlock = builder.createBlock( // FIXME : bad
  //        &f.getBody(), {}, TypeRange(f.getArgumentTypes()), argLocations);
  //

  //    entryBlock = builder.createBlock( // FIXME: bad
  //        &f.getBody().back(), TypeRange(f.getArgumentTypes()), argLocations);

  functionRegion = &f.getBody();

  assert(functionRegion != nullptr);

  return f;
}

} // namespace rust_compiler

/// Add new block with 'argTypes' arguments and set the insertion point to the
/// end of it. The block is inserted at the provided insertion point of
/// 'parent'. `locs` contains the locations of the inserted arguments, and
/// should match the size of `argTypes`.
//  Block *createBlock(Region *parent, Region::iterator insertPt = {},
//                     TypeRange argTypes = std::nullopt,
//                     ArrayRef<Location> locs = std::nullopt);

/// Add new block with 'argTypes' arguments and set the insertion point to the
/// end of it. The block is placed before 'insertBefore'. `locs` contains the
/// locations of the inserted arguments, and should match the size of
/// `argTypes`.
//  Block *createBlock(Block *insertBefore, TypeRange argTypes = std::nullopt,
//                     ArrayRef<Location> locs = std::nullopt);
