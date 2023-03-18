#include "TypeChecking.h"

#include <cassert>

namespace rust_compiler::sema::type_checking {

TypeCheckContext *TypeCheckContext::get() {
  static TypeCheckContext *instance;
  if (instance == nullptr)
    instance = new TypeCheckContext();
  return instance;
}

void TypeCheckContext::checkCrate(std::shared_ptr<ast::Crate>) {
  assert(false && "to be done");
}

void TypeCheckContext::insertBuiltin(basic::NodeId nodeId,
                                     basic::NodeId reference,
                                     TyTy::BaseType *type) {
  nodeToTypeReference[reference] = nodeId;
  resolvedTypes[nodeId] = type;
  builtinTypes.push_back(std::unique_ptr<TyTy::BaseType>(type));
}

void checkCrate(tyctx::TyCtx *tcx) { assert(false && "to be implemented"); }

} // namespace rust_compiler::sema::type_checking
