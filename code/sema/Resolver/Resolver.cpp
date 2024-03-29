#include "Resolver.h"

#include "ADT/CanonicalPath.h"
#include "AST/AssociatedItem.h"
#include "AST/ConstantItem.h"
#include "AST/Enumeration.h"
#include "AST/Implementation.h"
#include "AST/PathExpression.h"
#include "AST/StaticItem.h"
#include "AST/VisItem.h"
#include "AST/Visiblity.h"
#include "Basic/Ids.h"
#include "Lexer/Identifier.h"
#include "Session/Session.h"

#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <optional>

using namespace rust_compiler::basic;
using namespace rust_compiler::adt;
using namespace rust_compiler::ast;
using namespace rust_compiler::sema::type_checking;

namespace rust_compiler::sema::resolver {

bool Rib::wasDeclDeclaredHere(basic::NodeId def) const {
  for (auto &it : declsWithinRib)
    if (it.first == def)
      return true;
  return false;
}

void Rib::insertName(const adt::CanonicalPath &path, basic::NodeId id,
                     Location loc, bool shadow, RibKind kind) {
  // llvm::errs() << "insertName: " << path.toString() << "\n";
  auto it = pathMappings.find(path);
  if (it != pathMappings.end() && !shadow)
    return;

  pathMappings[path] = id;
  reversePathMappings.insert({id, path});
  declsWithinRib.insert({id, loc});
  references[id] = {};
  declTypeMappings.insert({id, kind});
  // llvm::errs() << "insertedName: " << path.toString() << "\n";
}

void Rib::clearName(const adt::CanonicalPath &path, basic::NodeId id) {
  auto ii = pathMappings.find(path);
  if (ii != pathMappings.end())
    pathMappings.erase(ii);

  auto ij = reversePathMappings.find(id);
  if (ij != reversePathMappings.end())
    reversePathMappings.erase(ij);

  auto ik = declsWithinRib.find(id);
  if (ik != declsWithinRib.end())
    declsWithinRib.erase(ik);
}

Rib *Scope::peek() { return stack.back(); }

void Scope::push(NodeId id) { stack.push_back(new Rib(getCrateNum(), id)); }

Rib *Scope::pop() {
  Rib *r = peek();
  stack.pop_back();
  return r;
}

void Scope::insert(const adt::CanonicalPath &path, basic::NodeId id,
                   Location loc, RibKind kind) {
  peek()->insertName(path, id, loc, true /*shadow*/, kind);
}

void Resolver::pushNewNameRib(Rib *r) { nameRibs[r->getNodeId()] = r; }
void Resolver::pushNewTypeRib(Rib *r) { typeRibs[r->getNodeId()] = r; }
void Resolver::pushNewLabelRib(Rib *r) { labelRibs[r->getNodeId()] = r; }
void Resolver::pushNewMaroRib(Rib *r) { macroRibs[r->getNodeId()] = r; }

Resolver::Resolver() noexcept
    : tyCtx(rust_compiler::session::session->getTypeContext()),
      nameScope(Scope(tyCtx->getCurrentCrate())),
      typeScope(Scope(tyCtx->getCurrentCrate())),
      labelScope(Scope(tyCtx->getCurrentCrate())),
      macroScope(Scope(tyCtx->getCurrentCrate())) {}

void Resolver::resolveItemNoRecurse(std::shared_ptr<ast::Item> item,
                                    const adt::CanonicalPath &prefix,
                                    const adt::CanonicalPath &canonicalPrefix) {
  switch (item->getItemKind()) {
  case ItemKind::VisItem: {
    resolveVisItemNoRecurse(std::static_pointer_cast<VisItem>(item), prefix,
                            canonicalPrefix);
    break;
  }
  case ItemKind::MacroItem: {
    resolveMacroItemNoRecurse(std::static_pointer_cast<MacroItem>(item), prefix,
                              canonicalPrefix);
    break;
  }
  }
}

void Resolver::resolveItem(std::shared_ptr<ast::Item> item,
                           const adt::CanonicalPath &prefix,
                           const adt::CanonicalPath &canonicalPrefix) {
  switch (item->getItemKind()) {
  case ItemKind::VisItem: {
    resolveVisItem(std::static_pointer_cast<VisItem>(item), prefix,
                   canonicalPrefix);
    break;
  }
  case ItemKind::MacroItem: {
    resolveMacroItem(std::static_pointer_cast<MacroItem>(item), prefix,
                     canonicalPrefix);
    break;
  }
  }
}

void Resolver::resolveCrate(std::shared_ptr<ast::Crate> crate) {
  // lookup current crate name
  CrateNum cnum = tyCtx->getCurrentCrate();
  llvm::errs() << cnum << "\n";

  std::optional<std::string> crateName = tyCtx->getCrateName(cnum);
  assert(crateName.has_value());

  llvm::errs() << "resolve: crate name: " << crate->getCrateName() << "\n";

  // setup the ribs
  NodeId scopeNodeId = crate->getNodeId();
  getNameScope().push(scopeNodeId);
  getTypeScope().push(scopeNodeId);
  getLabelScope().push(scopeNodeId);
  pushNewNameRib(getNameScope().peek());
  pushNewTypeRib(getTypeScope().peek());
  pushNewLabelRib(getLabelScope().peek());

  // builtins: FIXME
  insertBuiltinTypes(getTypeScope().peek());

  // get the root segment
  NodeId crateId = crate->getNodeId();
  CanonicalPath cratePrefix =
      CanonicalPath::newSegment(crateId, Identifier(crate->getCrateName()));
  cratePrefix.setCrateNum(crate->getCrateNum());

  // setup a dummy crate node
  getNameScope().insert(
      CanonicalPath::newSegment(crate->getNodeId(), Identifier("__$$crate__")),
      crate->getNodeId(), Location::getEmptyLocation(), RibKind::Dummy);

  // setup the root scope
  pushNewModuleScope(scopeNodeId);

  // only gather top-level
  for (auto &item : crate->getItems())
    resolveItemNoRecurse(item, CanonicalPath::createEmpty(), cratePrefix);

  // recursive
  for (auto &item : crate->getItems()) {
    tyCtx->insertItem(item.get());
    switch (item->getItemKind()) {
    case ItemKind::VisItem: {
      resolveVisItem(std::static_pointer_cast<VisItem>(item),
                     CanonicalPath::createEmpty(), cratePrefix);
      break;
    }
    case ItemKind::MacroItem: {
      resolveMacroItem(std::static_pointer_cast<MacroItem>(item),
                       CanonicalPath::createEmpty(), cratePrefix);
      break;
    }
    }
  }

  // done
  popModuleScope();
}

void Resolver::resolveVisItem(std::shared_ptr<ast::VisItem> visItem,
                              const adt::CanonicalPath &prefix,
                              const adt::CanonicalPath &canonicalPrefix) {
  tyCtx->insertLocation(visItem->getNodeId(), visItem->getLocation());
  switch (visItem->getKind()) {
  case VisItemKind::Module: {
    std::shared_ptr<ast::Module> mod =
        std::static_pointer_cast<Module>(visItem);
    basic::NodeId modNodeId = mod->getNodeId();
    tyCtx->insertModule(mod.get());
    pushNewModuleScope(modNodeId);
    resolveModule(mod, prefix, canonicalPrefix);
    popModuleScope();
    break;
  }
  case VisItemKind::ExternCrate: {
    assert(false && "to be handled later");
    break;
  }
  case VisItemKind::UseDeclaration: {
    assert(false && "to be handled later");
    break;
  }
  case VisItemKind::Function: {
    resolveFunction(std::static_pointer_cast<Function>(visItem).get(), prefix,
                    canonicalPrefix);
    break;
  }
  case VisItemKind::TypeAlias: {
    resolveTypeAlias(std::static_pointer_cast<TypeAlias>(visItem).get(), prefix,
                     canonicalPrefix);
    break;
  }
  case VisItemKind::Struct: {
    resolveStructItem(std::static_pointer_cast<Struct>(visItem), prefix,
                      canonicalPrefix);
    break;
  }
  case VisItemKind::Enumeration: {
    resolveEnumerationItem(std::static_pointer_cast<Enumeration>(visItem),
                           prefix, canonicalPrefix);
    break;
  }
  case VisItemKind::Union: {
    resolveUnionItem(std::static_pointer_cast<Union>(visItem), prefix,
                     canonicalPrefix);
    break;
  }
  case VisItemKind::ConstantItem: {
    resolveConstantItem(std::static_pointer_cast<ConstantItem>(visItem), prefix,
                        canonicalPrefix);
    break;
  }
  case VisItemKind::StaticItem: {
    resolveStaticItem(std::static_pointer_cast<StaticItem>(visItem), prefix,
                      canonicalPrefix);
    break;
  }
  case VisItemKind::Trait: {
    resolveTraitItem(std::static_pointer_cast<Trait>(visItem), prefix,
                     canonicalPrefix);
    break;
  }
  case VisItemKind::Implementation: {
    resolveImplementation(
        std::static_pointer_cast<ast::Implementation>(visItem), prefix,
        canonicalPrefix);
    break;
  }
  case VisItemKind::ExternBlock: {
    assert(false && "to be handled later");
    break;
  }
  }
}

void Resolver::resolveImplementation(
    std::shared_ptr<ast::Implementation> impl, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  tyCtx->insertImplementation(impl->getNodeId(), impl.get());
  
  switch (impl->getKind()) {
  case ImplementationKind::InherentImpl: {
    resolveInherentImpl(std::static_pointer_cast<ast::InherentImpl>(impl),
                        prefix, canonicalPrefix);
    break;
  }
  case ImplementationKind::TraitImpl: {
    resolveTraitImpl(std::static_pointer_cast<ast::TraitImpl>(impl), prefix,
                     canonicalPrefix);
    break;
  }
  }
}

void Resolver::resolveMacroItem(std::shared_ptr<ast::MacroItem>,
                                const adt::CanonicalPath &prefix,
                                const adt::CanonicalPath &canonicalPrefix) {
  assert(false && "to be handled later");
}

void Resolver::resolveInherentImpl(std::shared_ptr<ast::InherentImpl> implBlock,
                                   const adt::CanonicalPath &prefix,
                                   const adt::CanonicalPath &canonicalPrefix) {
  NodeId scopeNodeId = implBlock->getNodeId();
  getNameScope().push(scopeNodeId);
  getTypeScope().push(scopeNodeId);
  pushNewNameRib(getNameScope().peek());
  pushNewTypeRib(getTypeScope().peek());

  resolveVisibility(implBlock->getVisibility());

  if (implBlock->hasGenericParams()) {
    GenericParams parms = implBlock->getGenericParams();
    resolveGenericParams(parms, prefix, canonicalPrefix);
  }

  if (implBlock->hasWhereClause()) {
    WhereClause wh = implBlock->getWhereClause();
    resolveWhereClause(wh, prefix, canonicalPrefix);
  }

  resolveType(implBlock->getType(), prefix, canonicalPrefix);

  // setup canonical paths

  std::optional<CanonicalPath> selfCPath =
    resolveTypeToCanonicalPath(implBlock->getType().get(), prefix, canonicalPrefix);
  assert(selfCPath.has_value());
  // std::string typeName = resolveTypeToString(implBlock->getType().get());

  CanonicalPath implType = *selfCPath;
  CanonicalPath implPrefix = prefix.append(implType);
  CanonicalPath cpath = CanonicalPath::createEmpty();

  if (canonicalPrefix.getSize() <= 1) {
    cpath = *selfCPath;
  } else {
    std::string segBuffer = "<impl " + (*selfCPath).asString() + ">";
    CanonicalPath seg = CanonicalPath::newSegment(implBlock->getNodeId(),
                                                  lexer::Identifier(segBuffer));
    cpath = canonicalPrefix.append(seg);
  }

  // FIXME

  CanonicalPath Self = CanonicalPath::getBigSelf(implBlock->getNodeId());
  //CanonicalPath self = CanonicalPath::getSmallSelf(implBlock->getNodeId());

  getTypeScope().insert(Self, implBlock->getType()->getNodeId(),
                        implBlock->getType()->getLocation());
//  getTypeScope().insert(self, implBlock->getType()->getNodeId(),
//                        implBlock->getType()->getLocation());

  for (auto &asso : implBlock->getAssociatedItems())
    resolveAssociatedItem(&asso, implPrefix, cpath, implBlock->getNodeId());

  getTypeScope().peek()->clearName(Self, implBlock->getType()->getNodeId());
  //getTypeScope().peek()->clearName(self, implBlock->getType()->getNodeId());

  getTypeScope().pop();
  getNameScope().pop();
}

void Resolver::resolveTraitImpl(std::shared_ptr<ast::TraitImpl> impl,
                                const adt::CanonicalPath &prefix,
                                const adt::CanonicalPath &canonicalPrefix) {
  NodeId scopeNodeId = impl->getNodeId();

  resolveVisibility(impl->getVisibility());
  getNameScope().push(scopeNodeId);
  getTypeScope().push(scopeNodeId);
  getLabelScope().push(scopeNodeId);
  pushNewNameRib(getNameScope().peek());
  pushNewTypeRib(getTypeScope().peek());
  pushNewLabelRib(getLabelScope().peek());

  if (impl->hasGenericParams())
    resolveGenericParams(impl->getGenericParams(), prefix, canonicalPrefix);

  if (impl->hasWhereClause())
    resolveWhereClause(impl->getWhereClause(), prefix, canonicalPrefix);

  std::optional<basic::NodeId> path =
      resolveType(impl->getTypePath(), prefix, canonicalPrefix);
  if (!path) {
    getNameScope().pop();
    getTypeScope().pop();
    getLabelScope().pop();
    return;
  }

  std::optional<basic::NodeId> type =
      resolveType(impl->getType(), prefix, canonicalPrefix);
  if (!type) {
    getNameScope().pop();
    getTypeScope().pop();
    getLabelScope().pop();
    return;
  }

  std::optional<adt::CanonicalPath> canonTraitType = resolveTypeToCanonicalPath(
      impl->getTypePath().get(), prefix, canonicalPrefix);
  assert(canonTraitType.has_value());

  std::optional<adt::CanonicalPath> canonImplType = resolveTypeToCanonicalPath(
      impl->getType().get(), prefix, canonicalPrefix);
  assert(canonImplType.has_value());

  CanonicalPath implTypeSegment = *canonImplType;
  CanonicalPath traitTypeSegment = *canonTraitType;

  CanonicalPath projection = CanonicalPath::traitImplProjectionSegment(
      impl->getNodeId(), traitTypeSegment, implTypeSegment);

  CanonicalPath implPrefix = prefix.append(projection);

  CanonicalPath canonicalProjection = CanonicalPath::traitImplProjectionSegment(
      impl->getNodeId(), *canonTraitType, *canonImplType);

  CanonicalPath cpath = CanonicalPath::createEmpty();

  if (canonicalPrefix.getSize() <= 1) {
    cpath = canonicalProjection;
  } else {
    std::string projectionString = canonicalProjection.asString();
    std::string segmentBuffer =
        "<impl " + projectionString.substr(1, projectionString.size() - 2) +
        ">";
    CanonicalPath segment =
        CanonicalPath::newSegment(impl->getNodeId(), Identifier(segmentBuffer));
    cpath = canonicalPrefix.append(segment);
  }

  CanonicalPath Self = CanonicalPath::getBigSelf(impl->getType()->getNodeId());
  getTypeScope().insert(Self, impl->getType()->getNodeId(),
                        impl->getType()->getLocation());

  for (AssociatedItem &asso : impl->getAssociatedItems())
    resolveAssociatedItem(&asso, prefix, canonicalPrefix, impl->getNodeId());

  Rib *r = getTypeScope().peek();
  r->clearName(Self, impl->getType()->getNodeId());

  getNameScope().pop();
  getTypeScope().pop();
  getLabelScope().pop();
}

void Resolver::resolveVisibility(std::optional<ast::Visibility> vis) {
  if (vis) {
    switch (vis->getKind()) {
    case VisibilityKind::Private: {
      break;
    }
    case VisibilityKind::Public: {
      break;
    }
    case VisibilityKind::PublicCrate: {
      break;
    }
    case VisibilityKind::PublicSelf: {
      break;
    }
    case VisibilityKind::PublicSuper: {
      break;
    }
    case VisibilityKind::PublicIn: {
      resolveSimplePath(vis->getPath());
      break;
    }
    }
  }
}

void Resolver::insertResolvedName(NodeId ref, NodeId def) {
  // llvm::errs() << "insertResolvedName: " << ref << "->" << def << "\n";
  resolvedNames[ref] = def;
  getNameScope().appendReferenceForDef(ref, def);
  insertCapturedItem(def);
  tyCtx->insertResolvedName(ref, def);
}

void Rib::appendReferenceForDef(basic::NodeId ref, basic::NodeId def) {
  references[def].insert(ref);
}

void Scope::appendReferenceForDef(basic::NodeId ref, basic::NodeId def) {
  for (auto &rib : stack) {
    if (rib->wasDeclDeclaredHere(def)) {
      rib->appendReferenceForDef(ref, def);
    }
    return;
  }
  assert(false);
}

bool Resolver::declNeedsCapture(basic::NodeId declRibNodeId,
                                basic::NodeId closureRibNodeId,
                                const Scope &scope) {
  for (const auto &rib : scope.getContext()) {
    if (rib->getNodeId() == closureRibNodeId)
      return false;
    if (rib->getNodeId() == declRibNodeId)
      return true;
  }

  return false;
}

// for closure expressions
void Resolver::insertCapturedItem(basic::NodeId id) {
  if (closureContext.empty())
    return;

  // we are in a closure

  auto &nameScope = getNameScope();

  std::optional<RibKind> type = getNameScope().lookupDeclType(id);
  if (!type)
    return;

  std::optional<Rib *> rib = getNameScope().lookupRibForDecl(id);
  assert(rib.has_value());

  NodeId declRibNodeId = (*rib)->getNodeId();

  for (auto &closureExprId : closureContext) {

    if (!declNeedsCapture(declRibNodeId, closureExprId, nameScope))
      continue;

    if (*type != RibKind::Variable) {
      // FIXME: it ought to be a variable?
      return;
    }

    auto it = closureCaptureMappings.find(closureExprId);
    if (it != closureCaptureMappings.end()) {
      it->second.insert(id);
      tyCtx->insertClosureCapture(closureExprId, id);
    }
  }
}

std::optional<RibKind> Scope::lookupDeclType(NodeId id) {
  for (auto &rib : stack) {
    if (rib->wasDeclDeclaredHere(id)) {
      std::optional<RibKind> type = rib->lookupDeclType(id);
      if (type)
        return type;
    }
  }
  return std::nullopt;
}

std::optional<RibKind> Rib::lookupDeclType(basic::NodeId def) {
  auto it = declTypeMappings.find(def);
  if (it == declTypeMappings.end())
    return std::nullopt;
  return it->second;
}

std::optional<Rib *> Scope::lookupRibForDecl(basic::NodeId id) {
  for (auto &rib : stack) {
    if (rib->wasDeclDeclaredHere(id))
      return rib;
  }
  return std::nullopt;
}

void Resolver::insertResolvedType(basic::NodeId refId, basic::NodeId defId) {
  resolvedTypes[refId] = defId;
  getTypeScope().appendReferenceForDef(refId, defId);
  tyCtx->insertResolvedType(refId, defId);
}

bool Scope::wasDeclDeclaredInCurrentScope(NodeId def) const {
  for (auto &rib : stack) {
    if (rib->wasDeclDeclaredHere(def))
      return true;
  }
  return false;
}

std::optional<basic::NodeId>
Resolver::lookupResolvedName(basic::NodeId nodeId) {
  auto it = resolvedNames.find(nodeId);
  if (it == resolvedNames.end())
    return std::nullopt;
  return it->second;
}

std::optional<basic::NodeId>
Resolver::lookupResolvedType(basic::NodeId nodeId) {
  auto it = resolvedTypes.find(nodeId);
  if (it == resolvedTypes.end())
    return std::nullopt;
  return it->second;
}

void Resolver::insertBuiltinTypes(Rib *r) {
  auto builtins = tyCtx->getBuiltinTypes();
  for (auto &builtin : builtins) {
    CanonicalPath builtinPath = CanonicalPath::newSegment(
        builtin.second->getNodeId(), Identifier(builtin.first));
    r->insertName(builtinPath, builtin.second->getNodeId(),
                  Location::getBuiltinLocation(), false, RibKind::Type);
  }
}

void Resolver::pushClosureContext(basic::NodeId id) {
  auto it = closureCaptureMappings.find(id);
  assert(it == closureCaptureMappings.end());

  closureCaptureMappings.insert({id, {}});
  closureContext.push_back(id);
}

void Resolver::popClosureContext() {
  assert(!closureContext.empty());
  closureContext.pop_back();
}

std::optional<basic::NodeId> Rib::lookupName(const adt::CanonicalPath &ident) {
  auto it = pathMappings.find(ident);
  //  llvm::errs() << "rib::lookupName: " << ident.asString() << " "
  //               << (it == pathMappings.end()) << "\n";
  if (it == pathMappings.end())
    return std::nullopt;

  return it->second;
}

std::optional<basic::NodeId> Scope::lookup(const adt::CanonicalPath &p) {
  // llvm::errs() << "Scope::lookup: " << p.asString() << "\n";

  for (auto &rib : stack) {
    std::optional<NodeId> result = rib->lookupName(p);
    if (result)
      return *result;
  }
  // llvm::errs() << "Scope::lookup: " << p.asString() << ": failed"
  //              << "\n";
  return std::nullopt;
}

void Scope::print() const {
  for (auto &rib : stack)
    rib->print();
}

void Rib::print() const {
  for (auto &p : pathMappings)
    llvm::errs() << p.first.asString() << "\n";
}

void Resolver::insertResolvedMisc(NodeId ref, NodeId def) {
  auto it = miscResolvedItems.find(ref);
  if (it->second != def)
    assert(it == miscResolvedItems.end());
  miscResolvedItems[ref] = def;
}

// std::vector<std::pair<std::string, ast::types::TypeExpression *>> &
// Resolver::getBuiltinTypes() {
//   return builtins;
// }

} // namespace rust_compiler::sema::resolver

// items: canoncialPath

// namespaces: types, macros, values, lifeftimes
