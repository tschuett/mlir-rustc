#include "AST/FunctionParam.h"
#include "CrateBuilder/CrateBuilder.h"

#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>

using namespace rust_compiler::ast;

namespace rust_compiler::crate_builder {

mlir::FunctionType CrateBuilder::getFunctionType(ast::Function *fun) {
  assert(false);

  llvm::SmallVector<mlir::Type> parameterType;
  FunctionParameters params = fun->getParams();
  std::vector<FunctionParam> parms = params.getParams();
  for (FunctionParam &parm : parms) {
    switch (parm.getKind()) {
    case FunctionParamKind::Pattern: {
      parameterType.push_back(getType(parm.getType().get()));
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

  mlir::TypeRange inputs = parameterType;
  mlir::TypeRange results = {returnType};

  return builder.getFunctionType(inputs, results);
}

/// FIXME set visibility: { sym_visibility = "public" }
void CrateBuilder::emitFunction(std::shared_ptr<ast::Function> f) {
  mlir::FunctionType funType = getFunctionType(f.get());

  mlir::func::FuncOp fun = builder.create<mlir::func::FuncOp>(
      getLocation(f->getLocation()), "foo", funType);

  // mlir::MLIRContext *ctx = fun.getContext();

  //mlir::SymbolTable::setSymbolVisibility(&fun,
  //                                       mlir::SymbolTable::Visibility::Public);
  //  fun.setAttr(mlir::SymbolTable::getVisibilityAttrName(),
  //              mlir::StringAttr::get("public", ctx));
  //
  assert(false);
  if (f->hasBody())
    emitBlockExpression(
        std::static_pointer_cast<ast::BlockExpression>(f->getBody()));
}

} // namespace rust_compiler::crate_builder
