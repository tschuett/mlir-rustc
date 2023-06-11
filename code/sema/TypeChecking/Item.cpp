#include "ADT/CanonicalPath.h"
#include "ADT/ScopedCanonicalPath.h"
#include "AST/AssociatedItem.h"
#include "AST/ConstantItem.h"
#include "AST/Function.h"
#include "AST/GenericParams.h"
#include "AST/Implementation.h"
#include "AST/InherentImpl.h"
#include "AST/Struct.h"
#include "AST/StructField.h"
#include "AST/Types/TypePath.h"
#include "AST/VisItem.h"
#include "Coercion.h"
#include "TyCtx/Substitutions.h"
#include "TyCtx/TraitReference.h"
#include "TyCtx/TyTy.h"
#include "TyCtx/TypeIdentity.h"
#include "TypeChecking.h"

#include <memory>
#include <optional>
#include <vector>

using namespace rust_compiler::ast;
using namespace rust_compiler::tyctx;
using namespace rust_compiler::adt;

namespace rust_compiler::sema::type_checking {

void TypeResolver::checkVisItem(std::shared_ptr<ast::VisItem> v) {
  switch (v->getKind()) {
  case VisItemKind::Module: {
    assert(false && "to be implemented");
  }
  case VisItemKind::ExternCrate: {
    assert(false && "to be implemented");
  }
  case VisItemKind::UseDeclaration: {
    assert(false && "to be implemented");
  }
  case VisItemKind::Function: {
    checkFunction(std::static_pointer_cast<Function>(v));
    break;
  }
  case VisItemKind::TypeAlias: {
    checkTypeAlias(std::static_pointer_cast<TypeAlias>(v).get());
    break;
  }
  case VisItemKind::Struct: {
    checkStruct(static_cast<ast::Struct *>(v.get()));
    break;
  }
  case VisItemKind::Enumeration: {
    assert(false && "to be implemented");
  }
  case VisItemKind::Union: {
    assert(false && "to be implemented");
  }
  case VisItemKind::ConstantItem: {
    checkConstantItem(std::static_pointer_cast<ConstantItem>(v).get());
    break;
  }
  case VisItemKind::StaticItem: {
    assert(false && "to be implemented");
  }
  case VisItemKind::Trait: {
    checkTrait(std::static_pointer_cast<Trait>(v).get());
    break;
  }
  case VisItemKind::Implementation: {
    checkImplementation(static_cast<Implementation *>(v.get()));
    break;
  }
  case VisItemKind::ExternBlock: {
    assert(false && "to be implemented");
  }
  }
}

void TypeResolver::checkMacroItem(std::shared_ptr<ast::MacroItem> v) {
  assert(false && "to be implemented");
}

void TypeResolver::checkStruct(ast::Struct *s) {
  switch (s->getKind()) {
  case StructKind::StructStruct2: {
    checkStructStruct(static_cast<StructStruct *>(s));
    break;
  }
  case StructKind::TupleStruct2: {
    checkTupleStruct(static_cast<TupleStruct *>(s));
    break;
  }
  }
}

void TypeResolver::checkStructStruct(ast::StructStruct *s) {

  std::vector<TyTy::SubstitutionParamMapping> substitutions;
  if (s->hasGenerics())
    checkGenericParams(s->getGenericParams(), substitutions);

  if (s->hasWhereClause())
    checkWhereClause(s->getWhereClause());

  std::vector<TyTy::StructFieldType *> fields;

  if (s->hasStructFields()) {
    for (StructField &field : s->getFields().getFields()) {
      TyTy::BaseType *fieldType = checkType(field.getType());
      TyTy::StructFieldType *strField =
          new TyTy::StructFieldType(field.getNodeId(), field.getIdentifier(),
                                    fieldType, field.getLocation());
      fields.push_back(strField);
      tcx->insertType(field.getIdentity(), fieldType);
    }
  }

  std::optional<adt::CanonicalPath> path =
      tcx->lookupCanonicalPath(s->getNodeId());
  assert(path.has_value());
  tyctx::TypeIdentity ident = {*path, s->getLocation()};

  std::vector<TyTy::VariantDef *> variants;

  variants.push_back(new TyTy::VariantDef(s->getNodeId(), s->getIdentifier(),
                                          ident, TyTy::VariantKind::Struct,
                                          nullptr, fields));

  // parse #[repr(X)]
  TyTy::BaseType *type =
      new TyTy::ADTType(s->getNodeId(), s->getIdentifier(), ident,
                        TyTy::ADTKind::StructStruct, variants, substitutions);

  tcx->insertType(s->getIdentity(), type);
}

void TypeResolver::checkTupleStruct(ast::TupleStruct *s) {
  std::vector<TyTy::SubstitutionParamMapping> substitutions;
  if (s->hasGenerics())
    checkGenericParams(s->getGenericParams(), substitutions);

  if (s->hasWhereClause())
    checkWhereClause(s->getWhereClause());

  std::vector<TyTy::StructFieldType *> fields;

  size_t idx = 0;
  if (s->hasTupleFields()) {
    for (TupleField &field : s->getTupleFields().getFields()) {
      TyTy::BaseType *fieldType = checkType(field.getType());
      TyTy::StructFieldType *fiel = new TyTy::StructFieldType(
          field.getNodeId(), Identifier(std::to_string(idx)), fieldType,
          field.getLocation());
      fields.push_back(fiel);
      tcx->insertType(field.getIdentity(), fieldType);
      ++idx;
    }
  }

  std::optional<CanonicalPath> path = tcx->lookupCanonicalPath(s->getNodeId());
  assert(path.has_value());
  tyctx::TypeIdentity ident = {*path, s->getLocation()};

  std::vector<TyTy::VariantDef *> variants;
  variants.push_back(new TyTy::VariantDef(s->getNodeId(), s->getName(), ident,
                                          TyTy::VariantKind::Tuple, nullptr,
                                          fields));

  // parse #[rept(X)]

  TyTy::BaseType *type =
      new TyTy::ADTType(s->getNodeId(), s->getName(), ident,
                        TyTy::ADTKind::TupleStruct, variants, substitutions);

  tcx->insertType(s->getIdentity(), type);
}

void TypeResolver::checkImplementation(ast::Implementation *impl) {
  switch (impl->getKind()) {
  case ImplementationKind::InherentImpl: {
    checkInherentImpl(static_cast<InherentImpl *>(impl));
    break;
  }
  case ImplementationKind::TraitImpl: {
    checkTraitImpl(static_cast<TraitImpl *>(impl));
  }
  }
}

void TypeResolver::checkInherentImpl(ast::InherentImpl *impl) {
  std::optional<std::vector<TyTy::SubstitutionParamMapping>> substitutions =
      resolveInherentImplSubstitutions(impl);
  if (!substitutions) {
    assert(false);
  }

  TyTy::BaseType *self = resolveInherentImplSelf(impl);

  for (AssociatedItem &asso : impl->getAssociatedItems())
    checkImplementationItem(impl, asso, self, *substitutions);

  validateInherentImplBlock(impl, self, *substitutions);
}

void TypeResolver::checkTraitImpl(ast::TraitImpl *impl) { assert(false); }

void TypeResolver::checkConstantItem(ast::ConstantItem *con) {
  TyTy::BaseType *type = checkType(con->getType());
  if (con->hasInit()) {
    TyTy::BaseType *exprType = checkExpression(con->getInit());

    TyTy::BaseType *result = coercionWithSite(
        con->getNodeId(),
        TyTy::WithLocation(type, con->getType()->getLocation()),
        TyTy::WithLocation(exprType, con->getInit()->getLocation()),
        con->getLocation(), tcx);

    tcx->insertType(con->getIdentity(), result);
  }
}

void TypeResolver::checkTypeAlias(ast::TypeAlias *alias) {
  if (alias->hasType()) {
    TyTy::BaseType *actualType = checkType(alias->getType());

    tcx->insertType(alias->getIdentity(), actualType);
  }

  if (alias->hasWhereClause())
    checkWhereClause(alias->getWhereClause());
}

std::optional<std::vector<TyTy::SubstitutionParamMapping>>
TypeResolver::resolveInherentImplSubstitutions(InherentImpl *impl) {
  std::vector<TyTy::SubstitutionParamMapping> substitutions;
  if (impl->hasGenericParams())
    checkGenericParams(impl->getGenericParams(), substitutions);

  if (impl->hasWhereClause())
    checkWhereClause(impl->getWhereClause());

  TyTy::BaseType *self = checkType(impl->getType());
  if (self->getKind() == TyTy::TypeKind::Error)
    return std::nullopt;

  return substitutions;
}

std::optional<std::vector<TyTy::SubstitutionParamMapping>>
TypeResolver::resolveTraitImplSubstitutions(TraitImpl *impl) {
  std::vector<TyTy::SubstitutionParamMapping> substitutions;
  if (impl->hasGenericParams())
    checkGenericParams(impl->getGenericParams(), substitutions);

  if (impl->hasWhereClause())
    checkWhereClause(impl->getWhereClause());

  std::optional<TraitReference *> traitRef = resolveTraitPath(
      std::static_pointer_cast<ast::types::TypePath>(impl->getTypePath()));
  if (!traitRef)
    return std::nullopt;

  TyTy::TypeBoundPredicate specifiedBound =
      getPredicateFromBound(impl->getTypePath(), impl->getType().get());

  TyTy::BaseType *self = checkType(impl->getType());

  if (!specifiedBound.isError())
    self->inheritBounds({specifiedBound});

  const TyTy::SubstitutionArgumentMappings traitConstraints =
      specifiedBound.getSubstitutionArguments();
  const TyTy::SubstitutionArgumentMappings implConstraints =
      getUsedSubstitutionArguments(self);

  bool success = checkForUnconstrained(substitutions, traitConstraints,
                                       implConstraints, self);
  if (success)
    return substitutions;

  return std::nullopt;
}

bool TypeResolver::checkForUnconstrained(
    const std::vector<TyTy::SubstitutionParamMapping> &paramsToConstrain,
    const TyTy::SubstitutionArgumentMappings &constraintA,
    const TyTy::SubstitutionArgumentMappings &constraintB,
    const TyTy::BaseType *reference) {
  assert(false);
}

TyTy::BaseType *TypeResolver::resolveInherentImplSelf(InherentImpl *impl) {
  return checkType(impl->getType());
}

TyTy::BaseType *TypeResolver::resolveTraitImplSelf(InherentImpl *impl) {
  return checkType(impl->getType());
}

void TypeResolver::validateTraitImplBlock(
    TraitImpl *impl, TyTy::BaseType *self,
    std::vector<TyTy::SubstitutionParamMapping> &substitutions) {
  assert(false);
}

void TypeResolver::validateInherentImplBlock(
    InherentImpl *, TyTy::BaseType *self,
    std::vector<TyTy::SubstitutionParamMapping> &substitutions) {
  assert(false);
}

void TypeResolver::checkImplementationItem(
    ast::InherentImpl *impl, AssociatedItem &asso, TyTy::BaseType *self,
    std::vector<TyTy::SubstitutionParamMapping> substitutions) {
  assert(false);
}

} // namespace rust_compiler::sema::type_checking
