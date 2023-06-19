#include "ADT/CanonicalPath.h"
#include "AST/AssociatedItem.h"
#include "AST/EnumItemDiscriminant.h"
#include "AST/EnumItemStruct.h"
#include "AST/EnumItemTuple.h"
#include "AST/EnumItems.h"
#include "AST/Enumeration.h"
#include "AST/Expression.h"
#include "AST/Implementation.h"
#include "AST/InherentImpl.h"
#include "AST/StructStruct.h"
#include "AST/VisItem.h"
#include "Basic/Ids.h"
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
    resolveEnumerationNoRecurse(
        std::static_pointer_cast<Enumeration>(visItem).get(), prefix,
        canonicalPrefix);
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
    resolveImplementationNoRecurse(
        std::static_pointer_cast<Implementation>(visItem), prefix,
        canonicalPrefix);
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
    switch (asso.getKind()) {
    case AssociatedItemKind::MacroInvocationSemi: {
      assert(false);
    }
    case AssociatedItemKind::TypeAlias: {
      assert(false);
    }
    case AssociatedItemKind::ConstantItem: {
      assert(false);
    }
    case AssociatedItemKind::Function: {
      std::shared_ptr<Function> fun =
          std::static_pointer_cast<Function>(asso.getFunction());
      assert(fun->getKind() == VisItemKind::Function);
      assert((bool)fun);
      CanonicalPath decl =
          CanonicalPath::newSegment(fun->getNodeId(), fun->getName());
      CanonicalPath path2 = path.append(decl);
      getNameScope().insert(path2, fun->getNodeId(), fun->getLocation(),
                            RibKind::Function);
      break;
    }
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

void Resolver::resolveImplementationNoRecurse(
    std::shared_ptr<ast::Implementation> implementation,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  switch (implementation->getKind()) {
  case ImplementationKind::InherentImpl: {
    resolveInherentImplNoRecurse(
        std::static_pointer_cast<InherentImpl>(implementation), prefix,
        canonicalPrefix);
    break;
  }
  case ImplementationKind::TraitImpl: {
    resolveTraitImplNoRecurse(
        std::static_pointer_cast<TraitImpl>(implementation), prefix,
        canonicalPrefix);
    break;
  }
  }
}

void Resolver::resolveInherentImplNoRecurse(
    std::shared_ptr<ast::InherentImpl> implementation,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  for (AssociatedItem &asso : implementation->getAssociatedItems()) {
    if (asso.hasConstantItem()) {
      assert(false);
    } else if (asso.hasFunction()) {
      auto fun = std::static_pointer_cast<Function>(asso.getFunction());
      CanonicalPath segment =
          CanonicalPath::newSegment(fun->getNodeId(), fun->getName());
      CanonicalPath path = prefix.append(segment);
      getNameScope().insert(path, fun->getNodeId(), fun->getLocation(),
                            RibKind::Function);
    } else if (asso.hasMacroInvocationSemi()) {
      assert(false);
    } else if (asso.hasTypeAlias()) {
      assert(false);
    }
  }
}

void Resolver::resolveTraitImplNoRecurse(
    std::shared_ptr<ast::TraitImpl> implementation,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  for (AssociatedItem &asso : implementation->getAssociatedItems()) {
    if (asso.hasConstantItem()) {
      assert(false);
    } else if (asso.hasFunction()) {
      auto fun = std::static_pointer_cast<Function>(asso.getFunction());
      CanonicalPath segment =
          CanonicalPath::newSegment(fun->getNodeId(), fun->getName());
      CanonicalPath path = prefix.append(segment);
      getNameScope().insert(path, fun->getNodeId(), fun->getLocation(),
                            RibKind::Function);
    } else if (asso.hasMacroInvocationSemi()) {
      assert(false);
    } else if (asso.hasTypeAlias()) {
      assert(false);
    }
  }
}

void Resolver::resolveEnumerationNoRecurse(
    ast::Enumeration *enu, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  CanonicalPath decl =
      CanonicalPath::newSegment(enu->getNodeId(), enu->getName());
  CanonicalPath path = prefix.append(decl);
  CanonicalPath cpath = canonicalPrefix.append(decl);

  getTypeScope().insert(path, enu->getNodeId(), enu->getLocation(),
                        RibKind::Type);

  pushNewModuleScope(enu->getNodeId());

  if (enu->hasEnumItems()) {
    EnumItems it = enu->getEnumItems();
    for (const auto &en : it.getItems()) {

      if (en->hasStruct()) {
        EnumItemStruct str = en->getStruct();
        CanonicalPath decl2 =
            CanonicalPath::newSegment(str.getNodeId(), en->getName());
        CanonicalPath path2 = path.append(decl2);
        CanonicalPath cpath2 = cpath.append(decl2);
        getTypeScope().insert(path2, str.getNodeId(), str.getLocation(),
                              RibKind::Type);
        NodeId currentModule = peekCurrentModuleScope();
        tyCtx->insertCanonicalPath(str.getNodeId(), cpath2);
        tyCtx->insertModuleChildItem(str.getNodeId(), decl2);
        tyCtx->insertModuleChild(currentModule, str.getNodeId());
      } else if (en->hasTuple()) {
        EnumItemTuple str = en->getTuple();
        CanonicalPath decl2 =
            CanonicalPath::newSegment(str.getNodeId(), en->getName());
        CanonicalPath path2 = path.append(decl2);
        CanonicalPath cpath2 = cpath.append(decl2);
        getTypeScope().insert(path2, str.getNodeId(), str.getLocation(),
                              RibKind::Type);
        NodeId currentModule = peekCurrentModuleScope();
        tyCtx->insertCanonicalPath(str.getNodeId(), cpath2);
        tyCtx->insertModuleChildItem(str.getNodeId(), decl2);
        tyCtx->insertModuleChild(currentModule, str.getNodeId());
      }

      //if (en->hasDiscriminant()) {
      //  EnumItemDiscriminant str = en->getDiscriminant();
      //  CanonicalPath decl2 =
      //      CanonicalPath::newSegment(str.getNodeId(), en->getName());
      //  CanonicalPath path2 = path.append(decl2);
      //  CanonicalPath cpath2 = cpath.append(decl2);
      //  getTypeScope().insert(path2, str.getNodeId(), str.getLocation(),
      //                        RibKind::Type);
      //  NodeId currentModule = peekCurrentModuleScope();
      //  tyCtx->insertCanonicalPath(str.getNodeId(), cpath2);
      //  tyCtx->insertModuleChildItem(str.getNodeId(), decl2);
      //  tyCtx->insertModuleChild(currentModule, str.getNodeId());
      //}

      if (!en->hasStruct() and !en->hasTuple() and !en->hasDiscriminant()) {
        CanonicalPath decl =
            CanonicalPath::newSegment(en->getNodeId(), en->getName());

        CanonicalPath pathEn = path.append(decl);
        CanonicalPath cpathEn = cpath.append(decl);

        tyCtx->insertCanonicalPath(en->getNodeId(), cpathEn);
      }
    }
  }

  popModuleScope();

  NodeId currentModule = peekCurrentModuleScope();
  tyCtx->insertModuleChildItem(currentModule, decl);
  tyCtx->insertCanonicalPath(enu->getNodeId(), cpath);
}

} // namespace rust_compiler::sema::resolver
