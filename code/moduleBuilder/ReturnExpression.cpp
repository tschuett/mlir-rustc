#include "ModuleBuilder/ModuleBuilder.h"

#include <llvm/Support/raw_ostream.h>

using namespace llvm;

namespace rust_compiler {

void ModuleBuilder::emitReturnExpression(
    std::shared_ptr<ast::ReturnExpression> ret) {

  if (ret->getExpression()) {
    std::shared_ptr<ast::Expression> expr = ret->getExpression();
    mlir::Value mlirExpr = emitExpression(ret->getExpression());
    builder.create<mlir::func::ReturnOp>(getLocation(ret->getLocation()),
                                         ArrayRef(mlirExpr));
  } else {

    llvm::outs() << "returnOp w/o expr"
                 << "\n";

    builder.create<mlir::func::ReturnOp>(getLocation(ret->getLocation()));
  }
}

} // namespace rust_compiler
