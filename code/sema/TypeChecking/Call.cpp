#include "ADT/CanonicalPath.h"
#include "AST/MethodCallExpression.h"
#include "TyCtx/TypeIdentity.h"
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

  TypeIdentity ident = receiver->getTypeIdentity();
  adt::CanonicalPath path = ident.getPath();

  // reset
  methodCallCandidates.clear();

  methodCallReceiver = adt;
  methodCallFunQuery = method->getPath().getIdent().getIdentifier();
  tcx->iterateAssociatedItems([&](NodeId id, ast::Implementation *item,
                                  ast::AssociatedItem *impl) mutable -> bool {
    collectMethodCallCandidates(id, item, impl);
    return true;
  });

  llvm::errs() << "found candidates: " << methodCallCandidates.size() << "\n";
  for (auto &candi : methodCallCandidates) {
    llvm::errs() << candi.impl->getNodeId() << ": "
                 << candi.fun->getName().toString() << "\n";
  }
  //assert(methodCallCandidates.size() == 1);

  methodCallCandidates[0].fun->getNodeId();

  std::optional<TyTy::BaseType *> candidate =
      queryType(methodCallCandidates[0].fun->getNodeId());
  assert(candidate.has_value());

  return *candidate;

  // FIXME: reset
  // FIXME: visibility
  // FIXME: no generics!
}

void TypeResolver::collectMethodCallCandidates(NodeId id,
                                               ast::Implementation *item,
                                               ast::AssociatedItem *impl) {
  NodeId implTypeId;
  switch (item->getKind()) {
  case ImplementationKind::InherentImpl: {
    implTypeId = static_cast<ast::InherentImpl *>(item)->getType()->getNodeId();
    break;
  }
  case ImplementationKind::TraitImpl: {
    implTypeId = static_cast<ast::TraitImpl *>(item)->getType()->getNodeId();
    break;
  }
  }

  // FIXME: receiver is ADT with no traits
  std::optional<TyTy::BaseType *> implBlockType = queryType(implTypeId);
  if (!implBlockType) {
    llvm::errs() << "queryType failed: " << implTypeId << "\n";
    return;
  }
//
//  llvm::errs() << "check equality: " << (*implBlockType)->toString() << "\n";
//  llvm::errs() << "check equality: " << methodCallReceiver->toString() << "\n";
//  if (!methodCallReceiver->canEqual(*implBlockType, false))
//    if (!((*implBlockType)->canEqual(methodCallReceiver, false)))
//      return;

  if (methodCallReceiver->getTypeIdentity().getPath() !=
      (*implBlockType)->getTypeIdentity().getPath())
    return;

  // FIXME: item->visiit(this);
  switch (item->getKind()) {
  case ImplementationKind::InherentImpl: {
    auto *inherent = static_cast<ast::InherentImpl *>(item);
    for (AssociatedItem &asso : inherent->getAssociatedItems()) {
      if (asso.getKind() == ast::AssociatedItemKind::Function) {
        auto fun = static_cast<Function *>(
            static_cast<VisItem *>(asso.getFunction().get()));
        lexer::Identifier name = fun->getName();
        if (name == methodCallFunQuery) {
          methodCallCandidates.push_back({fun, item});
        }
      }
    }
    break;
  }
  case ImplementationKind::TraitImpl: {
    auto *trait = static_cast<ast::TraitImpl *>(item);
    for (AssociatedItem &asso : trait->getAssociatedItems()) {
      if (asso.getKind() == ast::AssociatedItemKind::Function) {
        auto fun = static_cast<Function *>(
            static_cast<VisItem *>(asso.getFunction().get()));
        lexer::Identifier name = fun->getName();
        if (name == methodCallFunQuery) {
          methodCallCandidates.push_back({fun, item});
        }
      }
    }
    break;
  }
  }
}

} // namespace rust_compiler::sema::type_checking
