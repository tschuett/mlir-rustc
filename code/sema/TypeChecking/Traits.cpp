#include "AST/VisItem.h"
#include "TyCtx/TraitReference.h"
#include "TypeChecking.h"
#include "mlir/IR/OpDefinition.h"

using namespace rust_compiler::tyctx;
using namespace rust_compiler::ast;

namespace rust_compiler::sema::type_checking {

TraitReference *
TypeResolver::resolveTraitPath(std::shared_ptr<ast::types::TypePath> path) {
  std::optional<Trait *> resolvedTraitReference = resolvePathToTrait(path);

  if (!resolvedTraitReference)
    return &TraitReference::errorNode();

  return resolveTrait(*resolvedTraitReference);
}

std::optional<Trait *>
TypeResolver::resolvePathToTrait(std::shared_ptr<ast::types::TypePath> path) {
  std::optional<basic::NodeId> ref = tcx->lookupResolvedType(path->getNodeId());
  if (!ref) {
    // report error
    assert(false);
  }

  std::optional<Item *> trait = tcx->lookupItem(*ref);
  if (!trait) {
    // report error
    assert(false);
  }

  assert((*trait)->getItemKind() == ItemKind::VisItem);
  VisItem *visItem = static_cast<VisItem*>(*trait);
  assert(visItem->getKind() == VisItemKind::Trait);
  Trait *t = static_cast<Trait*>(visItem);
  return t;
}

} // namespace rust_compiler::sema::type_checking
