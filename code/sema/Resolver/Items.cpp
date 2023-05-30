#include "ADT/CanonicalPath.h"
#include "AST/ConstantItem.h"
#include "AST/EnumItem.h"
#include "AST/EnumItemDiscriminant.h"
#include "AST/EnumItems.h"
#include "AST/Enumeration.h"
#include "AST/GenericParam.h"
#include "AST/Item.h"
#include "AST/MacroInvocationSemiItem.h"
#include "AST/TupleFields.h"
#include "AST/TypeAlias.h"
#include "AST/Types/TypeParamBounds.h"
#include "Basic/Ids.h"
#include "Lexer/Identifier.h"
#include "Resolver.h"

#include <memory>

using namespace rust_compiler::adt;
using namespace rust_compiler::ast;
using namespace rust_compiler::basic;

namespace rust_compiler::sema::resolver {

void Resolver::resolveModule(std::shared_ptr<ast::Module> mod,
                             const adt::CanonicalPath &prefix,
                             const adt::CanonicalPath &canonicalPrefix) {
  assert(false && "to be handled later");
  tyCtx->insertModule(mod.get());
}

void Resolver::resolveStaticItem(std::shared_ptr<ast::StaticItem> stat,
                                 const adt::CanonicalPath &prefix,
                                 const adt::CanonicalPath &canonicalPrefix) {
  CanonicalPath decl =
      CanonicalPath::newSegment(stat->getNodeId(), stat->getName());

  CanonicalPath path = prefix.append(decl);
  CanonicalPath cpath = canonicalPrefix.append(decl);

  tyCtx->insertCanonicalPath(stat->getNodeId(), cpath);

  resolveType(stat->getType(), prefix, canonicalPrefix);
  if (stat->hasInit())
    resolveExpression(stat->getInit(), path, cpath);
}

void Resolver::resolveConstantItem(std::shared_ptr<ast::ConstantItem> cons,
                                   const adt::CanonicalPath &prefix,
                                   const adt::CanonicalPath &canonicalPrefix) {
  CanonicalPath decl =
      CanonicalPath::newSegment(cons->getNodeId(), cons->getName());

  CanonicalPath path = prefix.append(decl);
  CanonicalPath cpath = canonicalPrefix.append(decl);

  // Bug fix
  getNameScope().insert(decl, cons->getNodeId(), cons->getLocation(),
                        RibKind::Variable);

  tyCtx->insertCanonicalPath(cons->getNodeId(), cpath);

  resolveVisibility(cons->getVisibility());

  resolveType(cons->getType(), prefix, canonicalPrefix);
  if (cons->hasInit())
    resolveExpression(cons->getInit(), path, cpath);
}

void Resolver::resolveEnumerationItem(
    std::shared_ptr<Enumeration> enu, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  assert(false && "to be handled later");
  CanonicalPath decl =
      CanonicalPath::newSegment(enu->getNodeId(), enu->getName());

  CanonicalPath path = prefix.append(decl);
  CanonicalPath cpath = canonicalPrefix.append(decl);

  tyCtx->insertCanonicalPath(enu->getNodeId(), cpath);
  tyCtx->insertEnumeration(enu->getNodeId(), enu.get());

  resolveVisibility(enu->getVisibility());

  NodeId scopeNodeId = enu->getNodeId();
  getTypeScope().push(scopeNodeId);

  if (enu->hasGenericParams())
    resolveGenericParams(enu->getGenericParams(), prefix, cpath);

  if (enu->hasWhereClause())
    resolveWhereClause(enu->getWhereClause(), prefix, canonicalPrefix);

  if (enu->hasEnumItems()) {
    std::vector<std::shared_ptr<EnumItem>> it = enu->getEnumItems().getItems();
    for (const auto &i : it) {
      resolveEnumItem(i, path, cpath);
      tyCtx->insertEnumItem(enu.get(), i.get());
    }
  }

  getTypeScope().pop();
}

void Resolver::resolveEnumItem(std::shared_ptr<ast::EnumItem> enuIt,
                               const adt::CanonicalPath &prefix,
                               const adt::CanonicalPath &canonicalPrefix) {

  switch (enuIt->getKind()) {
  case EnumItemKind::Discriminant: {
    EnumItemDiscriminant dis = enuIt->getDiscriminant();
    CanonicalPath decl =
        CanonicalPath::newSegment(dis.getNodeId(), enuIt->getName());
    CanonicalPath path = prefix.append(decl);
    CanonicalPath cpath = canonicalPrefix.append(decl);
    tyCtx->insertCanonicalPath(dis.getNodeId(), cpath);
    break;
  }
  case EnumItemKind::Struct: {
    EnumItemStruct str = enuIt->getStruct();
    CanonicalPath decl =
        CanonicalPath::newSegment(str.getNodeId(), enuIt->getName());
    CanonicalPath path = prefix.append(decl);
    CanonicalPath cpath = canonicalPrefix.append(decl);
    tyCtx->insertCanonicalPath(str.getNodeId(), cpath);

    if (str.hasFields()) {
      std::vector<StructField> fields = str.getFields().getFields();
      for (const StructField &field : fields)
        resolveType(field.getType(), prefix, canonicalPrefix);
    }
    break;
  }
  case EnumItemKind::Tuple: {
    EnumItemTuple tup = enuIt->getTuple();
    CanonicalPath decl =
        CanonicalPath::newSegment(tup.getNodeId(), enuIt->getName());
    CanonicalPath path = prefix.append(decl);
    CanonicalPath cpath = canonicalPrefix.append(decl);
    tyCtx->insertCanonicalPath(tup.getNodeId(), cpath);
    if (tup.hasTupleFiels()) {
      std::vector<TupleField> fields = tup.getTupleFields().getFields();
      for (const TupleField &tup : fields)
        resolveType(tup.getType(), prefix, canonicalPrefix);
    }
    break;
  }
  }
}

void Resolver::resolveTraitItem(std::shared_ptr<ast::Trait> trait,
                                const adt::CanonicalPath &prefix,
                                const adt::CanonicalPath &canonicalPrefix) {
  NodeId scopeNodeId = trait->getNodeId();

  CanonicalPath segment =
      CanonicalPath::newSegment(trait->getNodeId(), trait->getIdentifier());
  CanonicalPath path = prefix.append(segment);
  CanonicalPath cpath = canonicalPrefix.append(segment);

  tyCtx->insertCanonicalPath(trait->getNodeId(), cpath);

  // Bug fix. Note that it is before the push pop pair!
  getTypeScope().insert(segment, trait->getNodeId(), trait->getLocation(),
                        RibKind::Trait);

  resolveVisibility(trait->getVisibility());

  getNameScope().push(scopeNodeId);
  getTypeScope().push(scopeNodeId);
  pushNewNameRib(getNameScope().peek());
  pushNewTypeRib(getTypeScope().peek());

  TypeParam param = TypeParam(trait->getLocation());
  param.setIdentifier(lexer::Identifier("Self"));

  GenericParam gp = {trait->getLocation()};
  gp.setTypeParam(param);
  trait->insertImplicitSelf(gp);

  CanonicalPath Self = CanonicalPath::getBigSelf(trait->getNodeId());

  if (trait->hasGenericParams())
    resolveGenericParams(trait->getGenericParams(), prefix, canonicalPrefix);

  getTypeScope().appendReferenceForDef(Self.getNodeId(), param.getNodeId());

  if (trait->hasTypeParamBounds())
    for (auto &b : trait->getTypeParamBounds().getBounds())
      resolveTypeParamBound(b, prefix, canonicalPrefix);

  if (trait->hasWhereClause())
    resolveWhereClause(trait->getWhereClause(), prefix, canonicalPrefix);

  CanonicalPath path2 = CanonicalPath::createEmpty();
  CanonicalPath cpath2 = CanonicalPath::createEmpty();

  for (auto &asso : trait->getAssociatedItems())
    resolveAssociatedItemInTrait(asso, path2, cpath2);

  getTypeScope().pop();
  getNameScope().pop();
}

void Resolver::resolveAssociatedItem(
    const ast::AssociatedItem &asso, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {

  resolveVisibility(asso.getVisibility());

  if (asso.hasTypeAlias()) {
    assert(false);
    // resolveAssociatedTypeAlias(asso.getTypeAlias(), prefix, canonicalPrefix);
  } else if (asso.hasConstantItem()) {
    assert(false);
    // resolveAssociatedConstantItem(asso.getConstantItem(), prefix,
    //                               canonicalPrefix);
  } else if (asso.hasFunction()) {
    resolveAssociatedFunction(
        static_cast<Function *>(
            std::static_pointer_cast<VisItem>(asso.getFunction()).get()),
        prefix, canonicalPrefix);
  } else if (asso.hasMacroInvocationSemi()) {
    assert(false);
    // resolveAssociatedMacroInvocationSemi(asso.getMacroItem(), prefix,
    //                                      canonicalPrefix);
  }

  // FIXME function is method!!!! only in type checking!!!
}

void Resolver::resolveAssociatedFunction(
    ast::Function *fun, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  resolveFunction(fun, prefix, canonicalPrefix);
}

void Resolver::resolveAssociatedTypeAlias(
    ast::TypeAlias *, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  assert(false);
}

void Resolver::resolveAssociatedConstantItem(
    ast::ConstantItem *, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  assert(false);
}

void Resolver::resolveAssociatedMacroInvocationSemi(
    ast::MacroInvocationSemiItem *, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  assert(false);
}

void Resolver::resolveAssociatedItemInTrait(
    const ast::AssociatedItem &asso, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  llvm::errs() << "resolveAssociatedItem"
               << "\n";
  llvm::errs() << asso.hasMacroInvocationSemi() << "\n";
  llvm::errs() << asso.hasTypeAlias() << "\n";
  llvm::errs() << asso.hasConstantItem() << "\n";
  llvm::errs() << asso.hasFunction() << "\n";
  if (asso.hasMacroInvocationSemi())
    resolveMacroInvocationSemiInTrait(
        static_cast<MacroInvocationSemiItem *>(asso.getMacroItem().get()),
        prefix, canonicalPrefix);
  else if (asso.hasTypeAlias())
    resolveTypeAliasInTrait(static_cast<TypeAlias *>(asso.getTypeAlias().get()),
                            prefix, canonicalPrefix);
  else if (asso.hasConstantItem())
    resolveConstantItemInTrait(
        static_cast<ConstantItem *>(asso.getConstantItem().get()), prefix,
        canonicalPrefix);
  else if (asso.hasFunction())
    resolveFunctionInTrait(static_cast<Function *>(asso.getFunction().get()),
                           prefix, canonicalPrefix);
}

void Resolver::resolveUnionItem(std::shared_ptr<ast::Union> uni,
                                const adt::CanonicalPath &prefix,
                                const adt::CanonicalPath &canonicalPrefix) {
  CanonicalPath decl =
      CanonicalPath::newSegment(uni->getNodeId(), uni->getIdentifier());
  CanonicalPath path = prefix.append(decl);
  CanonicalPath cpath = canonicalPrefix.append(decl);

  tyCtx->insertCanonicalPath(uni->getNodeId(), cpath);

  resolveVisibility(uni->getVisibility());
  basic::NodeId scopeNodeId = uni->getNodeId();
  getTypeScope().push(scopeNodeId);

  if (uni->hasGenericParams())
    resolveGenericParams(uni->getGenericParams(), prefix, canonicalPrefix);

  if (uni->hasWhereClause())
    resolveWhereClause(uni->getWhereClause(), prefix, canonicalPrefix);

  StructFields fields = uni->getStructFields();
  for (StructField &field : fields.getFields())
    resolveType(field.getType(), prefix, canonicalPrefix);

  getTypeScope().pop();
}

void Resolver::resolveTypeAlias(ast::TypeAlias *alias,
                                const adt::CanonicalPath &prefix,
                                const adt::CanonicalPath &canonicalPrefix) {
  CanonicalPath decl =
      CanonicalPath::newSegment(alias->getNodeId(), alias->getIdentifier());
  CanonicalPath path = prefix.append(decl);
  CanonicalPath cpath = canonicalPrefix.append(decl);

  tyCtx->insertCanonicalPath(alias->getNodeId(), cpath);

  // Bug fix. Note that it is before the push pop pair!
  getTypeScope().insert(decl, alias->getNodeId(), alias->getLocation(),
                        RibKind::Type);

  NodeId scopeNodeId = alias->getNodeId();
  getTypeScope().push(scopeNodeId);

  if (alias->hasGenericParams())
    resolveGenericParams(alias->getGenericParams(), prefix, canonicalPrefix);

  if (alias->hasWhereClause())
    resolveWhereClause(alias->getWhereClause(), prefix, canonicalPrefix);

  if (alias->hasType())
    resolveType(alias->getType(), prefix, canonicalPrefix);

  getTypeScope().pop();
}

} // namespace rust_compiler::sema::resolver
