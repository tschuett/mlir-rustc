#pragma once

#include "ADT/CanonicalPath.h"
#include "ADT/ScopedHashTable.h"
#include "AST/AssociatedItem.h"
#include "AST/Types/TupleType.h"
#include "Basic/Ids.h"
#include "Sema/Autoderef.h"
#include "TyCtx/AssociatedImplTrait.h"
#include "TyCtx/NodeIdentity.h"
#include "TyCtx/TraitReference.h"
#include "TyCtx/TyTy.h"

#include <llvm/ADT/STLFunctionalExtras.h>
#include <map>
#include <optional>
#include <set>
#include <vector>

namespace rust_compiler::ast {
class Module;
class Item;
class Crate;
class Enumeration;
class EnumItem;
class ExternalItem;
class Implementation;
} // namespace rust_compiler::ast

namespace rust_compiler::sema::type_checking {
class TypeResolver;
};

namespace rust_compiler::tyctx {

using namespace rust_compiler::sema::type_checking;
using namespace rust_compiler::basic;

class TyCtx {
public:
  TyCtx();

  void iterateImplementations(
      llvm::function_ref<bool(NodeId, ast::Implementation *)> cb);
  void
  iterateAssociatedItems(llvm::function_ref<bool(NodeId, ast::Implementation *,
                                                 ast::AssociatedItem *)>
                             cb);
  void insertModule(ast::Module *);
  void insertItem(ast::Item *);
  void insertEnumeration(NodeId, ast::Enumeration *);
  /// to find the Enumeration of an EnumItem
  void insertEnumItem(ast::Enumeration *, ast::EnumItem *, NodeId id);
  void insertImplementation(NodeId, ast::Implementation *);
  void insertReceiver(NodeId, TyTy::BaseType *);
  void insertAssociatedItem(NodeId implementationId, ast::AssociatedItem *);

  ast::Module *lookupModule(basic::NodeId);

  std::optional<adt::CanonicalPath>
  lookupModuleChild(basic::NodeId module, const adt::CanonicalPath &path);
  std::optional<std::pair<ast::Enumeration *, ast::EnumItem *>>
  lookupEnumItem(NodeId id);
  void insertVariantDefinition(NodeId id, NodeId variant);

  void insertCanonicalPath(basic::NodeId id, const adt::CanonicalPath &path) {
    if (auto canPath = lookupCanonicalPath(id)) {
      if (canPath->isEqual(path))
        return;
      llvm::errs() << "surprise in insertCanonicalPath: " << id << "\n";
      llvm::errs() << "old: " << canPath->asString() << "\n";
      llvm::errs() << "new: " << path.asString() << "\n";
      assert(canPath->getSize() >= path.getSize());
    }
    paths.emplace(id, path);
  }

  void insertChildItemToParentModuleMapping(basic::NodeId child,
                                            basic::NodeId parentModule) {
    childToParentModuleMap.insert({child, parentModule});
  }

  std::optional<adt::CanonicalPath> lookupCanonicalPath(basic::NodeId id) {
    auto it = paths.find(id);
    if (it == paths.end())
      return std::nullopt;
    return it->second;
  }

  void insertModuleChildItem(basic::NodeId module,
                             const adt::CanonicalPath &child) {
    auto it = moduleChildItems.find(module);
    if (it == moduleChildItems.end())
      moduleChildItems.insert({module, {child}});
    else
      it->second.emplace_back(child);
  }

  void insertModuleChild(NodeId module, NodeId child) {
    auto it = moduleChildMap.find(module);
    if (it == moduleChildMap.end())
      moduleChildMap.insert({module, {child}});
    else
      it->second.emplace_back(child);
  }

  basic::CrateNum getCurrentCrate() const;
  void setCurrentCrate(basic::CrateNum);

  std::optional<std::string> getCrateName(CrateNum cnum);

  bool isModule(basic::NodeId query);
  bool isCrate(basic::NodeId query) const;
  // void insertNodeToNode(basic::NodeId id, basic::NodeId ref);

  std::optional<std::vector<adt::CanonicalPath>>
  lookupModuleChildrenItems(basic::NodeId module);

  void insertASTCrate(ast::Crate *crate, basic::CrateNum crateNum);

  void insertBuiltin(basic::NodeId id, basic::NodeId ref, TyTy::BaseType *type);
  TyTy::BaseType *lookupBuiltin(std::string_view name);

  void insertType(const NodeIdentity &id, TyTy::BaseType *type);

  void insertImplicitType(const basic::NodeId &id, TyTy::BaseType *type) {
    resolved[id] = type;
  }

  void insertAutoderefMapping(NodeId, std::vector<sema::Adjustment>);

  std::optional<TyTy::BaseType *> lookupType(basic::NodeId);
  std::optional<ast::Item *> lookupItem(basic::NodeId);
  std::optional<ast::ExternalItem *> lookupExternalItem(basic::NodeId);
  std::optional<ast::Implementation *> lookupImplementation(basic::NodeId);
  // return implementationId and AssociatedItem for AssociatedItemId
  std::optional<std::pair<NodeId, ast::AssociatedItem *>>
      lookupAssociatedItem(basic::NodeId);
  std::optional<basic::NodeId> lookupVariantDefinition(basic::NodeId);
  std::optional<NodeId> lookupAssociatedTypeMapping(NodeId id);

  [[nodiscard]] std::optional<basic::NodeId> lookupName(basic::NodeId);

  void insertResolvedName(basic::NodeId ref, basic::NodeId def);
  void insertResolvedType(basic::NodeId ref, basic::NodeId def);
  std::optional<basic::NodeId> lookupResolvedName(basic::NodeId);
  std::optional<basic::NodeId> lookupResolvedType(basic::NodeId);

  std::vector<std::pair<std::string, ast::types::TypeExpression *>> &
  getBuiltinTypes() {
    return builtins;
  }

  std::optional<TyTy::TypeBoundPredicate> lookupPredicate(NodeId);
  void insertResolvedPredicate(basic::NodeId id,
                               TyTy::TypeBoundPredicate predicate);

  void insertClosureCapture(basic::NodeId closureExpr,
                            basic::NodeId capturedItem);

  std::set<basic::NodeId> getCaptures(NodeId);

  Location lookupLocation(basic::NodeId);
  void insertLocation(basic::NodeId, Location);

  void insertOperatorOverLoad(basic::NodeId id, TyTy::FunctionType *callSite);

  // traits
  void insertTraitQuery(basic::NodeId id);
  void traitQueryCompleted(basic::NodeId id);
  std::optional<TyTy::TraitReference *> lookupTraitReference(basic::NodeId id);
  bool isTraitQueryInProgress(basic::NodeId id) const;
  void insertTraitReference(basic::NodeId id, TyTy::TraitReference &&ref);

  void insertAssociatedTypeMapping(basic::NodeId id, basic::NodeId mapping);

  void pushNewIteratorLoopContext(basic::NodeId, Location loc);
  TyTy::BaseType *popLoopContext();
  TyTy::BaseType *peekLoopContext() const;

  void insertAssociatedTraitImpl(NodeId id, AssociatedImplTrait &&associated);
  std::optional<AssociatedImplTrait *> lookupAssociatedTraitImpl(NodeId id);

  void insertAssociatedImplMapping(NodeId traitId,
                                   const TyTy::BaseType *impl_Type,
                                   NodeId implId);
  std::optional<NodeId>
  lookupAssociatedImplMappingForSelf(NodeId traitId,
                                     const TyTy::BaseType *self);

  bool haveCheckedForUnconstrained(NodeId id, bool *result);
  void insertUnconstrainedCheckMarker(NodeId id, bool status);

private:
  void generateBuiltins();

  void setupBuiltin(std::string_view name, TyTy::BaseType *tyty);
  void setUnitTypeNodeId(basic::NodeId id) { unitTyNodeId = id; }

  std::map<NodeId, bool> unconstrained;

  // basic::CrateNum crateNumIter = 7;
  // basic::NodeId nodeIdIter = 7;
  basic::CrateNum currentCrateNum = basic::UNKNOWN_CREATENUM;

  basic::NodeId unitTyNodeId = UNKNOWN_NODEID;
  basic::NodeId globalTypeNodeId = basic::UNKNOWN_NODEID;

  std::map<basic::NodeId, ast::Module *> modules;
  std::map<basic::NodeId, ast::Item *> items;
  std::map<basic::NodeId, adt::CanonicalPath> paths;

  // Maps each module's node id to a list of its children
  std::map<basic::NodeId, std::vector<basic::NodeId>> moduleChildMap;
  std::map<basic::NodeId, std::vector<adt::CanonicalPath>> moduleChildItems;
  std::map<basic::NodeId, basic::NodeId> childToParentModuleMap;

  std::map<basic::CrateNum, ast::Crate *> astCrateMappings;

  std::map<basic::NodeId, basic::NodeId> nodeIdRefs;
  std::map<basic::NodeId, TyTy::BaseType *> resolved;
  std::vector<std::unique_ptr<TyTy::BaseType>> builtinsList;

  std::map<basic::NodeId, basic::NodeId> resolvedNames;
  std::map<basic::NodeId, basic::NodeId> resolvedTypes;

  std::map<NodeId, std::pair<ast::Enumeration *, ast::EnumItem *>>
      enumItemsMappings;
  std::map<NodeId, ast::Enumeration *> enumMappings;

  std::map<NodeId, ast::Item *> itemMappings;
  std::map<NodeId, ast::Implementation *> implementationMappings;
  // associatedItem id -> {implementationId, AssociatedItem}
  std::map<NodeId, std::pair<NodeId, ast::AssociatedItem *>>
      associatedItemMappings;
  std::map<NodeId, std::pair<ast::ExternalItem *, NodeId>> externItemMappings;
  std::map<NodeId, std::pair<NodeId, ast::Implementation *>>
      hirImplItemMappings;

  std::map<NodeId, std::vector<sema::Adjustment>> autoderefMappings;

  // std::map<NodeId, std::vector<NodeId>> moduleChildMap;

  // TyTy

  std::unique_ptr<TyTy::UintType> u8;
  std::unique_ptr<TyTy::UintType> u16;
  std::unique_ptr<TyTy::UintType> u32;
  std::unique_ptr<TyTy::UintType> u64;
  std::unique_ptr<TyTy::UintType> u128;

  std::unique_ptr<TyTy::IntType> i8;
  std::unique_ptr<TyTy::IntType> i16;
  std::unique_ptr<TyTy::IntType> i32;
  std::unique_ptr<TyTy::IntType> i64;
  std::unique_ptr<TyTy::IntType> i128;

  std::unique_ptr<TyTy::FloatType> f32;
  std::unique_ptr<TyTy::FloatType> f64;

  std::unique_ptr<TyTy::BoolType> rbool;

  std::unique_ptr<TyTy::USizeType> usize;
  std::unique_ptr<TyTy::ISizeType> isize;

  std::unique_ptr<TyTy::CharType> charType;
  std::unique_ptr<TyTy::StrType> strType;
  std::unique_ptr<TyTy::NeverType> never;

  std::vector<std::pair<std::string, ast::types::TypeExpression *>> builtins;

  ast::types::TupleType *emptyTupleType;

  // closure captures
  std::map<basic::NodeId, std::set<basic::NodeId>> closureCaptureMappings;

  std::map<NodeId, NodeId> associatedTypeMappings;
  std::map<NodeId, AssociatedImplTrait> associatedImplTraits;

  std::map<NodeId, Location> locations;
  std::map<basic::NodeId, basic::NodeId> variants;
  std::map<basic::NodeId, TyTy::BaseType *> receiverContext;
  std::map<basic::NodeId, TyTy::FunctionType *> operatorOverloads;

  std::set<basic::NodeId> traitQueriesInProgress;
  std::map<basic::NodeId, TyTy::TraitReference> traitContext;

  std::map<basic::NodeId, TyTy::TypeBoundPredicate> predicates;

  std::vector<TyTy::BaseType *> loopTypeStack;
};

} // namespace rust_compiler::tyctx
