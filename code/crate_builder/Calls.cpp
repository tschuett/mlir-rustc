#include "CrateBuilder/CrateBuilder.h"
#include "TyCtx/TyCtx.h"
#include "TyCtx/TyTy.h"
#include "TyCtx/TypeIdentity.h"

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::tyctx::TyTy;
using namespace rust_compiler::tyctx;

namespace rust_compiler::crate_builder {

mlir::Value CrateBuilder::emitCallExpression(ast::CallExpression *expr) {
  llvm::errs() << "emitCallExpression; " << expr->getLocation().toString()
               << "\n";
  std::optional<tyctx::TyTy::BaseType *> type =
      tyCtx->lookupType(expr->getFunction()->getNodeId());
  if (type) {
    llvm::errs() << "  " << (*type)->toString() << "\n";
    if ((*type)->getKind() == TypeKind::ADT) {
      TyTy::ADTType *adt = static_cast<TyTy::ADTType *>(*type);
      if (adt->getKind() == ADTKind::Enum)
        return emitTupleStructConstructor(expr);
    }

    TypeIdentity identity = (*type)->getTypeIdentity();
    llvm::errs() << identity.getPath().asString() << "\n";
  }
  llvm::errs() << ""
               << "\n";
  // What is the return type?
  assert(false);
}

mlir::Value
CrateBuilder::emitMethodCallExpression(ast::MethodCallExpression *expr) {
  assert(false);
}

mlir::Value
CrateBuilder::emitTupleStructConstructor(ast::CallExpression *expr) {
  llvm::errs() << "emitTupleStructExpression; " << expr->getLocation().toString()
               << "\n";
  std::optional<tyctx::TyTy::BaseType *> type =
      tyCtx->lookupType(expr->getFunction()->getNodeId());
  if (type) {
    TypeIdentity identity = (*type)->getTypeIdentity();
    llvm::errs() << identity.getPath().asString() << "\n";
  }
  assert(false);
}

} // namespace rust_compiler::crate_builder
