#include "AST/MethodCallExpression.h"
#include "TypeChecking.h"

#include <cstdlib>
#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::tyctx;

namespace rust_compiler::sema::type_checking {

/// https://rustc-dev-guide.rust-lang.org/method-lookup.html
TyTy::BaseType *TypeResolver::checkMethodCallExpression(
    TyTy::FunctionType *, NodeIdentity, std::vector<TyTy::Argument> &args,
    Location call, Location receiver, TyTy::BaseType *adjustedSelf) {
  assert(false);
}

TyTy::BaseType *
TypeResolver::checkMethodCallExpression(ast::MethodCallExpression *method) {
  TyTy::BaseType *receiver = checkExpression(method->getReceiver());
  tcx->insertReceiver(method->getNodeId(), receiver);

  if (receiver->getKind() != TyTy::TypeKind::ADT) {
    llvm::errs() << "receiver kind not supported"
                 << "\n";
    exit(EXIT_FAILURE);
  }

  TyTy::ADTType *adt = static_cast<TyTy::ADTType *>(receiver);

  if (adt->getKind() != TyTy::ADTKind::StructStruct) {
    llvm::errs() << "receiver kind not supported"
                 << "\n";
    exit(EXIT_FAILURE);
  }

  assert(false);
}

} // namespace rust_compiler::sema::type_checking
