#include "ADT/CanonicalPath.h"
#include "AST/StructStruct.h"
#include "Resolver.h"

#include <memory>

using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace rust_compiler::basic;
using namespace rust_compiler::sema;

namespace rust_compiler::sema::resolver {

void Resolver::resolveVisItemNoRecurse(
    std::shared_ptr<ast::VisItem> visItem, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {

  NodeId currentModule = peekCurrentModuleScope();
  tyCtx->insertChildItemToParentModuleMapping(visItem->getNodeId(),
                                              currentModule);
  switch (visItem->getKind()) {
  case VisItemKind::Module: {
    assert(false);
    break;
  }
  case VisItemKind::ExternCrate: {
    assert(false);
    break;
  }
  case VisItemKind::UseDeclaration: {
    assert(false);
    break;
  }
  case VisItemKind::Function: {
    resolveFunctionNoRecurse(std::static_pointer_cast<Function>(visItem),
                             prefix, canonicalPrefix);
    break;
  }
  case VisItemKind::TypeAlias: {
    resolveTypeAliasNoRecurse(
        std::static_pointer_cast<TypeAlias>(visItem).get(), prefix,
        canonicalPrefix);
    break;
  }
  case VisItemKind::Struct: {
    resolveStructNoRecurse(std::static_pointer_cast<Struct>(visItem).get(),
                           prefix, canonicalPrefix);
    break;
  }
  case VisItemKind::Enumeration: {
    assert(false);
    break;
  }
  case VisItemKind::Union: {
    assert(false);
    break;
  }
  case VisItemKind::ConstantItem: {
    assert(false);
    break;
  }
  case VisItemKind::StaticItem: {
    assert(false);
    break;
  }
  case VisItemKind::Trait: {
    resolveTraitNoRecurse(std::static_pointer_cast<Trait>(visItem), prefix,
                          canonicalPrefix);
    break;
  }
  case VisItemKind::Implementation: {
    assert(false);
    break;
  }
  case VisItemKind::ExternBlock: {
    assert(false);
    break;
  }
  }
}

void Resolver::resolveFunctionNoRecurse(
    std::shared_ptr<ast::Function> fun, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  CanonicalPath segment =
      CanonicalPath::newSegment(fun->getNodeId(), fun->getName());
  CanonicalPath path = canonicalPrefix.append(segment);

  /// FIXME
  getNameScope().insert(path, fun->getNodeId(), fun->getLocation(),
                        RibKind::Function);

  NodeId currentModule = peekCurrentModuleScope();
  tyCtx->insertModuleChildItem(currentModule, segment);
  tyCtx->insertCanonicalPath(fun->getNodeId(), path);
}

void Resolver::resolveMacroItemNoRecurse(
    std::shared_ptr<ast::MacroItem>, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  assert(false && "to be implemented");
}

void Resolver::resolveTraitNoRecurse(
    std::shared_ptr<ast::Trait> trait, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  CanonicalPath segment =
      CanonicalPath::newSegment(trait->getNodeId(), trait->getIdentifier());
  CanonicalPath path = canonicalPrefix.append(segment);

  /// FIXME
  getTypeScope().insert(path, trait->getNodeId(), trait->getLocation(),
                        RibKind::Trait);

  for (auto &asso : trait->getAssociatedItems()) {
    if (asso.hasFunction()) {
      auto fun = std::static_pointer_cast<Function>(asso.getFunction());
      CanonicalPath decl =
          CanonicalPath::newSegment(fun->getNodeId(), fun->getName());
      CanonicalPath path2 = path.append(decl);
      getNameScope().insert(path2, fun->getNodeId(), fun->getLocation(),
                            RibKind::Function);
    } else if (asso.hasTypeAlias()) {
      assert(false);
    } else if (asso.hasConstantItem()) {
      assert(false);
    } else if (asso.hasMacroInvocationSemi()) {
      assert(false);
    }
  }

  NodeId currentModule = peekCurrentModuleScope();
  tyCtx->insertModuleChildItem(currentModule, segment);
  tyCtx->insertCanonicalPath(trait->getNodeId(), path);
}

void Resolver::resolveStructNoRecurse(
    ast::Struct *str, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  switch (str->getKind()) {
  case StructKind::StructStruct2: {
    resolveStructStructNoRecurse(static_cast<ast::StructStruct *>(str), prefix,
                                 canonicalPrefix);
    break;
  }
  case StructKind::TupleStruct2: {
    resolveTupleStructNoRecurse(static_cast<ast::TupleStruct *>(str), prefix,
                                canonicalPrefix);
    break;
  }
  }
}

void Resolver::resolveStructStructNoRecurse(
    ast::StructStruct *str, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  CanonicalPath segment =
      CanonicalPath::newSegment(str->getNodeId(), str->getIdentifier());
  CanonicalPath path = prefix.append(segment);
  CanonicalPath cpath = canonicalPrefix.append(segment);

  /// FIXME
  getTypeScope().insert(path, str->getNodeId(), str->getLocation(),
                        RibKind::Type);

  NodeId currentModule = peekCurrentModuleScope();
  tyCtx->insertModuleChildItem(currentModule, segment);
  tyCtx->insertCanonicalPath(str->getNodeId(), cpath);
}

void Resolver::resolveTupleStructNoRecurse(
    ast::TupleStruct *str, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  CanonicalPath segment =
      CanonicalPath::newSegment(str->getNodeId(), str->getName());
  CanonicalPath path = prefix.append(segment);
  CanonicalPath cpath = canonicalPrefix.append(segment);

  /// FIXME
  getTypeScope().insert(path, str->getNodeId(), str->getLocation(),
                        RibKind::Type);

  NodeId currentModule = peekCurrentModuleScope();
  tyCtx->insertModuleChildItem(currentModule, segment);
  tyCtx->insertCanonicalPath(str->getNodeId(), cpath);
}

void Resolver::resolveTypeAliasNoRecurse(
    ast::TypeAlias *alias, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  CanonicalPath segment =
      CanonicalPath::newSegment(alias->getNodeId(), alias->getIdentifier());
  CanonicalPath path = prefix.append(segment);
  CanonicalPath cpath = canonicalPrefix.append(segment);

  /// FIXME
  getTypeScope().insert(path, alias->getNodeId(), alias->getLocation(),
                        RibKind::Type);

  NodeId currentModule = peekCurrentModuleScope();
  tyCtx->insertModuleChildItem(currentModule, segment);
  tyCtx->insertCanonicalPath(alias->getNodeId(), cpath);
}

} // namespace rust_compiler::sema::resolver
