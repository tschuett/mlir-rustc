#include "AST/FunctionParam.h"
#include "Basic/Ids.h"
#include "CrateBuilder/CrateBuilder.h"
#include "mlir/IR/Location.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>

using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

namespace rust_compiler::crate_builder {

void CrateBuilder::declare(basic::NodeId id, mlir::Value val) {

  llvm::errs() << "declare: " << id << "\n";

  symbolTable.insert(id, val);
}

mlir::FunctionType CrateBuilder::getFunctionType(ast::Function *fun) {
  llvm::SmallVector<mlir::Type> parameterType;
  FunctionParameters params = fun->getParams();
  std::vector<FunctionParam> parms = params.getParams();
  for (FunctionParam &parm : parms) {
    switch (parm.getKind()) {
    case FunctionParamKind::Pattern: {
      FunctionParamPattern pattern = parm.getPattern();
      assert(pattern.hasType());
      parameterType.push_back(getType(pattern.getType().get()));
      break;
    }
    case FunctionParamKind::DotDotDot: {
      assert(false && "to be implemented later");
    }
    case FunctionParamKind::Type: {
      assert(false && "to be implemented later");
    }
    }
  }

  mlir::Type returnType = builder.getNoneType();

  if (fun->hasReturnType())
    returnType = getType(fun->getReturnType().get());

  llvm::errs() << "getFunctionType: " << parameterType.size() << "\n";

  mlir::TypeRange inputs = parameterType;
  mlir::TypeRange results = {returnType};

  return builder.getFunctionType(inputs, results);
}

/// FIXME set visibility: { sym_visibility = "public" }
void CrateBuilder::emitFunction(ast::Function *f) {

  llvm::ScopedHashTableScope<basic::NodeId, mlir::Value> scope(symbolTable);

  builder.setInsertionPointToEnd(theModule.getBody());

  mlir::FunctionType funType = getFunctionType(f);

  llvm::errs() << "inputs: " << funType.getNumInputs() << "\n";

  llvm::errs() << "outputs: " << funType.getNumResults() << "\n";

  llvm::SmallVector<basic::NodeId> parameterNames;
  llvm::SmallVector<mlir::Type> parameterTypes;
  llvm::SmallVector<mlir::Location> parameterLocation;
  for (FunctionParam &parm : f->getParams().getParams()) {
    if (parm.getKind() == FunctionParamKind::Pattern) {
      parameterNames.push_back(parm.getPattern().getPattern()->getNodeId());
      llvm::errs() << "pattern: " << parm.getPattern().getPattern()->getNodeId()
                   << "\n";
      parameterTypes.push_back(getType(parm.getPattern().getType().get()));
      parameterLocation.push_back(getLocation(parm.getLocation()));
    } else {
      assert(false);
    }
  }

  mlir::func::FuncOp fun = builder.create<mlir::func::FuncOp>(
      getLocation(f->getLocation()), "foo", funType);
  assert(fun != nullptr);

  llvm::errs() << "declaration?: " << fun.isDeclaration() << "\n";

  llvm::errs() << "block arguments: " << parameterTypes.size() << "\n";

  // mlir::Block* entryBlock = builder.createBlock(&fun.front(),
  // parameterTypes);
  mlir::Block *entryBlock = builder.createBlock(
      &fun.getBody(), {}, parameterTypes, parameterLocation);

  // mlir::Block &entryBlock = fun.front();

  llvm::errs() << parameterNames.size() << "\n";
  llvm::errs() << "PROBLEM: " << entryBlock->getArguments().size() << "\n";
  llvm::errs() << "PROBLEM: " << entryBlock->getNumArguments() << "\n";

  for (const auto namedValue :
       llvm::zip(parameterNames, entryBlock->getArguments()))
    declare(std::get<0>(namedValue), std::get<1>(namedValue));

  llvm::errs() << symbolTable.count(109) << "\n";
  llvm::errs() << symbolTable.count(parameterNames[0]) << "\n";

  // mlir::MLIRContext *ctx = fun.getContext();

  // mlir::SymbolTable::setSymbolVisibility(&fun,
  //                                        mlir::SymbolTable::Visibility::Public);
  //   fun.setAttr(mlir::SymbolTable::getVisibilityAttrName(),
  //               mlir::StringAttr::get("public", ctx));
  //

  builder.setInsertionPointToStart(entryBlock);

  if (f->hasBody())
    emitBlockExpression(
        static_cast<ast::BlockExpression *>(f->getBody().get()));

  functionMap.insert({"foo", fun});
}

} // namespace rust_compiler::crate_builder
