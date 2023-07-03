#include "ADT/CanonicalPath.h"
#include "ADT/ScopedCanonicalPath.h"
#include "AST/AssociatedItem.h"
#include "AST/ConstantItem.h"
#include "AST/EnumItemDiscriminant.h"
#include "AST/EnumItemStruct.h"
#include "AST/EnumItemTuple.h"
#include "AST/Enumeration.h"
#include "AST/Function.h"
#include "AST/FunctionParameters.h"
#include "AST/GenericParams.h"
#include "AST/Implementation.h"
#include "AST/InherentImpl.h"
#include "AST/LiteralExpression.h"
#include "AST/Patterns/IdentifierPattern.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/SelfParam.h"
#include "AST/ShorthandSelf.h"
#include "AST/Struct.h"
#include "AST/StructField.h"
#include "AST/StructFields.h"
#include "AST/TraitImpl.h"
#include "AST/TypedSelf.h"
#include "AST/Types/TypePath.h"
#include "AST/VisItem.h"
#include "Basic/Ids.h"
#include "Coercion.h"
#include "Lexer/Identifier.h"
#include "Session/Session.h"
#include "TyCtx/NodeIdentity.h"
#include "TyCtx/Substitutions.h"
#include "TyCtx/TraitReference.h"
#include "TyCtx/TyTy.h"
#include "TyCtx/TypeIdentity.h"
#include "TyCtx/Unification.h"
#include "TypeChecking.h"

#include <cstdlib>
#include <limits>
#include <llvm/Support/raw_ostream.h>
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
    checkEnumeration(static_cast<ast::Enumeration *>(v.get()));
    break;
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
      if (fieldType == nullptr ||
          fieldType->getKind() == TyTy::TypeKind::Error) {
        llvm::errs() << "checkStructStruct failed @"
                     << s->getLocation().toString() << "\n";
        exit(EXIT_FAILURE);
      }
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
  llvm::errs() << "checkInherentImpl"
               << "\n";
  std::optional<std::vector<TyTy::SubstitutionParamMapping>> substitutions =
      resolveInherentImplSubstitutions(impl);
  if (!substitutions) {
    assert(false);
  }

  TyTy::BaseType *type = checkType(impl->getType());
  tcx->insertType(impl->getType()->getIdentity(), type);

  TyTy::BaseType *self = resolveInherentImplSelf(impl);

  for (AssociatedItem &asso : impl->getAssociatedItems())
    checkInherentImplItem(impl, asso, self, *substitutions);

  validateInherentImplBlock(impl, self, *substitutions);
}

void TypeResolver::checkTraitImpl(ast::TraitImpl *impl) {
  std::optional<std::vector<TyTy::SubstitutionParamMapping>> substitutions =
      resolveTraitImplSubstitutions(impl);
  if (!substitutions) {
    assert(false);
  }

  TyTy::BaseType *type = checkType(impl->getType());
  tcx->insertType(impl->getType()->getIdentity(), type);

  TyTy::BaseType *typePath = checkType(impl->getTypePath());
  tcx->insertType(impl->getTypePath()->getIdentity(), typePath);

  TyTy::BaseType *self = resolveTraitImplSelf(impl);

  for (AssociatedItem &asso : impl->getAssociatedItems())
    checkTraitImplItem(impl, asso, self, *substitutions);

  validateTraitImplBlock(impl, self, *substitutions);
}

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

  std::optional<TyTy::TraitReference *> traitRef = resolveTraitPath(
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

  bool checkResult = false;
  bool checkCompleted =
      tcx->haveCheckedForUnconstrained(reference->getReference(), &checkResult);
  if (checkCompleted)
    return checkResult;

  std::set<basic::NodeId> symbolsToContrain;
  std::map<NodeId, Location> symbolToLocation;
  for (const auto &p : paramsToConstrain) {
    NodeId ref = p.getParamType()->getReference();
    symbolsToContrain.insert(ref);
    symbolToLocation.insert({ref, p.getParamLocation()});
  }

  std::set<basic::NodeId> constrainedSymbols;
  for (const auto &c : constraintA.getMappings()) {
    TyTy::BaseType *arg = c.getType();
    if (arg != nullptr) {
      const TyTy::BaseType *p = arg->getRoot();
      constrainedSymbols.insert(p->getTypeReference());
    }
  }

  for (const auto &c : constraintB.getMappings()) {
    TyTy::BaseType *arg = c.getType();
    if (arg != nullptr) {
      const TyTy::BaseType *p = arg->getRoot();
      constrainedSymbols.insert(p->getTypeReference());
    }
  }

  const TyTy::BaseType *root = reference->getRoot();
  if (root->getKind() == TyTy::TypeKind::Parameter) {
    const TyTy::ParamType *p = static_cast<const TyTy::ParamType *>(root);
    constrainedSymbols.insert(p->getTypeReference());
  }

  bool unconstrained = false;
  for (auto &sym : symbolsToContrain) {
    if (constrainedSymbols.find(sym) == constrainedSymbols.end()) {
      Location loc = symbolToLocation.at(sym);
      unconstrained = true;
    }
  }

  tcx->insertUnconstrainedCheckMarker(reference->getReference(), unconstrained);

  return unconstrained;
}

TyTy::BaseType *TypeResolver::resolveInherentImplSelf(InherentImpl *impl) {
  return checkType(impl->getType());
}

TyTy::BaseType *TypeResolver::resolveTraitImplSelf(TraitImpl *impl) {
  return checkType(impl->getType());
}

void TypeResolver::validateTraitImplBlock(
    TraitImpl *impl, TyTy::BaseType *self,
    std::vector<TyTy::SubstitutionParamMapping> &substitutions) {
  assert(false);

  //  if (is_trait_impl_block) {
  //    trait_reference->clear_associated_types();

  //    AssociatedImplTrait associated(trait_reference, specified_bound,
  //                                   &impl_block, self, context);
  //    tcx->insertAssociatedTraitImpl(impl_block.get_mappings().get_hirid(),
  //                                          std::move(associated));
  //    tcx->insertAssociatedImplMapping(
  //        trait_reference->get_mappings().get_hirid(), self,
  //        impl_block.get_mappings().get_hirid());
  //  }
}

void TypeResolver::validateInherentImplBlock(
    InherentImpl *impl, TyTy::BaseType *self,
    std::vector<TyTy::SubstitutionParamMapping> &substitutions) {}

void TypeResolver::checkTraitImplItem(
    ast::TraitImpl *impl, AssociatedItem &asso, TyTy::BaseType *self,
    std::vector<TyTy::SubstitutionParamMapping> substitutions) {
  assert(false);
}

void TypeResolver::checkInherentImplItem(
    ast::InherentImpl *impl, AssociatedItem &asso, TyTy::BaseType *self,
    std::vector<TyTy::SubstitutionParamMapping> substitutions) {
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
    checkImplementationFunction(impl,
                                static_cast<Function *>(static_cast<VisItem *>(
                                    asso.getFunction().get())),
                                self, substitutions);
    break;
  }
  }
}

TyTy::BaseType *TypeResolver::checkImplementationFunction(
    ast::TraitImpl *parent, ast::Function *fun, TyTy::BaseType *self,
    std::vector<TyTy::SubstitutionParamMapping> substitutions) {
  assert(false);
}

TyTy::BaseType *TypeResolver::checkImplementationFunction(
    ast::InherentImpl *parent, ast::Function *fun, TyTy::BaseType *self,
    std::vector<TyTy::SubstitutionParamMapping> substitutions) {
  if (fun->hasGenericParams())
    checkGenericParams(fun->getGenericParams(), substitutions);

  if (fun->hasWhereClause())
    checkWhereClause(fun->getWhereClause());
  TyTy::BaseType *returnType = nullptr;
  if (!fun->hasReturnType()) {
    returnType = TyTy::TupleType::getUnitType(fun->getNodeId());
  } else {
    TyTy::BaseType *resolved = checkType(fun->getReturnType());
    if (resolved == nullptr) {
      llvm::errs() << fun->getLocation().toString()
                   << "@ failed to resolved return type"
                   << "\n";
      exit(EXIT_FAILURE);
    }
    returnType = resolved->clone();
    returnType->setReference(fun->getReturnType()->getNodeId());
  }

  std::vector<
      std::pair<std::shared_ptr<patterns::PatternNoTopAlt>, TyTy::BaseType *>>
      params;
  if (fun->isMethod()) {
    SelfParam selfParam = fun->getParams().getSelfParam();
    std::shared_ptr<patterns::IdentifierPattern> selfPattern =
        std::make_shared<patterns::IdentifierPattern>(
            patterns::IdentifierPattern(self->getLocation()));
    selfPattern->setIdentifier(lexer::Identifier("self"));
    // FIXME
    // selfPattern->setRef();
    // selfPattern->setRef()
    TyTy::BaseType *selfType = nullptr;
    switch (selfParam.getKind()) {
    case SelfParamKind::ShorthandSelf: {
      ShorthandSelf *hand =
          std::static_pointer_cast<ShorthandSelf>(selfParam.getSelf()).get();
      if (hand->isAnd())
        selfPattern->setRef();
      if (hand->isMut())
        selfPattern->setMut();
      if ((!hand->isMut() && !hand->isAnd()) || hand->isMut()) {
        selfType = self->clone();
      } else if (hand->isAnd() && !hand->isMut()) {
        selfType = new TyTy::ReferenceType(
            selfParam.getNodeId(), TyTy::TypeVariable(self->getReference()),
            Mutability::Imm);
      } else if (hand->isAnd() && hand->isMut()) {
        selfType = new TyTy::ReferenceType(
            selfParam.getNodeId(), TyTy::TypeVariable(self->getReference()),
            Mutability::Mut);
      }
      break;
    }
    case SelfParamKind::TypeSelf: {
      TypedSelf *self =
          std::static_pointer_cast<TypedSelf>(selfParam.getSelf()).get();
      selfType = checkType(self->getType());
      if (self->isMut())
        selfPattern->setMut();
      break;
    }
    }
    tcx->insertType(selfParam.getIdentity(), selfType);
    params.push_back(
        std::pair<std::shared_ptr<patterns::PatternNoTopAlt>, TyTy::BaseType *>(
            selfPattern, selfType));
  }

  if (fun->hasParams()) {
    FunctionParameters parameters = fun->getParams();
    for (auto &param : parameters.getParams()) {
      TyTy::BaseType *paramType = checkType(param.getType());
      params.push_back(std::pair<std::shared_ptr<patterns::PatternNoTopAlt>,
                                 TyTy::BaseType *>(
          param.getPattern().getPattern().get(), paramType));
      tcx->insertType(param.getIdentity(), paramType);
      checkPattern(param.getPattern().getPattern(), paramType);
    }
  }

  std::optional<CanonicalPath> canon =
      tcx->lookupCanonicalPath(fun->getNodeId());
  assert(canon.has_value());

  TypeIdentity ident = {*canon, fun->getLocation()};

  TyTy::FunctionType *funType = new TyTy::FunctionType(
      fun->getNodeId(), fun->getName(), ident,
      fun->isMethod() ? TyTy::FunctionType::FunctionTypeIsMethod
                      : TyTy::FunctionType::FunctionTypeDefaultFlags,
      params, returnType, std::move(substitutions));

  tcx->insertType(fun->getIdentity(), funType);

  TyTy::FunctionType *resolvedFunType = funType;
  pushReturnType(TypeCheckContextItem(fun), resolvedFunType->getReturnType());

  if (fun->hasBody()) {

    pushSmallSelf(self);
    TyTy::BaseType *blockExprType = checkExpression(fun->getBody());
    popSmallSelf();

    Location funReturnLoc = fun->hasReturnType()
                                ? fun->getReturnType()->getLocation()
                                : fun->getLocation();

    Coercion coerce = {tcx};
    coerce.coercion(funType->getReturnType(), blockExprType,
                    fun->getBody()->getLocation(), false /*allowAutoderef*/);
  }

  popReturnType();

  return funType;
}

void TypeResolver::checkEnumeration(ast::Enumeration *enu) {
  std::vector<TyTy::SubstitutionParamMapping> substitutions;
  if (enu->hasGenericParams())
    checkGenericParams(enu->getGenericParams(), substitutions);

  std::vector<TyTy::VariantDef *> variants;
  int64_t discriminantValue = 0;
  if (enu->hasEnumItems()) {
    EnumItems items = enu->getEnumItems();
    for (auto &item : items.getItems()) {
      TyTy::VariantDef *fieldType =
          checkEnumItem(item.get(), discriminantValue);

      ++discriminantValue;
      variants.push_back(fieldType);
    }
  }

  std::optional<CanonicalPath> path =
      tcx->lookupCanonicalPath(enu->getNodeId());
  assert(path.has_value());

  TypeIdentity ident = {*path, enu->getLocation()};

  TyTy::BaseType *type =
      new TyTy::ADTType(enu->getNodeId(), rust_compiler::basic::getNextNodeId(),
                        enu->getName(), ident, TyTy::ADTKind::Enum,
                        std::move(variants), std::move(substitutions));

  tcx->insertType(enu->getIdentity(), type);
}

TyTy::VariantDef *TypeResolver::checkEnumItem(EnumItem *enuItem,
                                              int64_t discriminant) {
  assert(discriminant < std::numeric_limits<int64_t>::max());

  if (enuItem->hasStruct())
    return checkEnumItemStruct(enuItem->getStruct(), enuItem->getName(),
                               discriminant);
  else if (enuItem->hasTuple())
    return checkEnumItemTuple(enuItem->getTuple(), enuItem->getName(),
                              discriminant);

  // FIXME
  //  if (enuItem->hasDiscriminant())
  //    return checkEnumItemDiscriminant(enuItem->getDiscriminant(),
  //                                     enuItem->getName(), discriminant);

  LiteralExpression *discrimExpr =
      new LiteralExpression(enuItem->getLocation());
  discrimExpr->setKind(LiteralExpressionKind::IntegerLiteral);
  discrimExpr->setStorage(std::to_string(discriminant));

  std::optional<TyTy::BaseType *> isize = tcx->lookupBuiltin("isize");
  assert(isize.has_value());
  tcx->insertType(enuItem->getIdentity(), *isize);

  std::optional<CanonicalPath> canon =
      tcx->lookupCanonicalPath(enuItem->getNodeId());
  if (!canon) {
    llvm::errs() << "checkEnumItem: failed to lookup canonical path for "
                 << enuItem->getNodeId() << "\n";
    exit(EXIT_FAILURE);
  }

  TypeIdentity ident = {*canon, enuItem->getLocation()};

  return new TyTy::VariantDef(enuItem->getNodeId(), enuItem->getName(), ident,
                              discrimExpr);
}

TyTy::VariantDef *TypeResolver::checkEnumItemTuple(const EnumItemTuple &enuItem,
                                                   const Identifier &name,
                                                   int64_t discriminant) {
  std::vector<TyTy::StructFieldType *> fields;
  size_t idx = 0;
  if (enuItem.hasTupleFields()) {
    TupleFields f = enuItem.getTupleFields();
    for (const TupleField &t : f.getFields()) {
      TyTy::BaseType *fieldType = checkType(t.getType());
      TyTy::StructFieldType *typeField = new TyTy::StructFieldType(
          t.getNodeId(), Identifier(std::to_string(idx)), fieldType,
          t.getLocation());
      fields.push_back(typeField);
      tcx->insertType(t.getIdentity(), typeField->getFieldType());
      ++idx;
    }
  }

  LiteralExpression *discrimExpr = new LiteralExpression(enuItem.getLocation());
  discrimExpr->setKind(LiteralExpressionKind::IntegerLiteral);
  discrimExpr->setStorage(std::to_string(discriminant));

  std::optional<TyTy::BaseType *> isize = tcx->lookupBuiltin("isize");
  assert(isize.has_value());
  tcx->insertType(enuItem.getIdentity(), *isize);

  std::optional<CanonicalPath> canon =
      tcx->lookupCanonicalPath(enuItem.getNodeId());
  assert(canon.has_value());

  TypeIdentity ident = {*canon, enuItem.getLocation()};

  return new TyTy::VariantDef(enuItem.getNodeId(), name, ident,
                              TyTy::VariantKind::Tuple, discrimExpr, fields);
}

TyTy::VariantDef *
TypeResolver::checkEnumItemStruct(const EnumItemStruct &enuItem,
                                  const Identifier &name,
                                  int64_t discriminant) {
  std::vector<TyTy::StructFieldType *> fields;
  if (enuItem.hasFields()) {
    StructFields filds = enuItem.getFields();
    for (const StructField &f : filds.getFields()) {
      TyTy::BaseType *fieldType = checkType(f.getType());
      TyTy::StructFieldType *typeOfField = new TyTy::StructFieldType(
          f.getNodeId(), f.getIdentifier(), fieldType, f.getLocation());
      fields.push_back(typeOfField);
      tcx->insertType(f.getIdentity(), typeOfField->getFieldType());
    }
  }

  LiteralExpression *discrimExpr = new LiteralExpression(enuItem.getLocation());
  discrimExpr->setKind(LiteralExpressionKind::IntegerLiteral);
  discrimExpr->setStorage(std::to_string(discriminant));

  std::optional<TyTy::BaseType *> isize = tcx->lookupBuiltin("isize");
  assert(isize.has_value());
  tcx->insertType(enuItem.getIdentity(), *isize);

  std::optional<CanonicalPath> path =
      tcx->lookupCanonicalPath(enuItem.getNodeId());
  assert(path.has_value());

  TypeIdentity ident = {*path, enuItem.getLocation()};

  return new TyTy::VariantDef(enuItem.getNodeId(), name, ident,
                              TyTy::VariantKind::Struct, discrimExpr, fields);
}

TyTy::VariantDef *
TypeResolver::checkEnumItemDiscriminant(const EnumItemDiscriminant &enuItem,
                                        const Identifier &name,
                                        int64_t discriminant) {
  TyTy::BaseType *capacityType = checkExpression(enuItem.getExpression());

  TyTy::ISizeType *expectedType =
      new TyTy::ISizeType(enuItem.getExpression()->getNodeId());
  tcx->insertType(enuItem.getExpression()->getIdentity(), expectedType);

  Unification::unifyWithSite(
      TyTy::WithLocation(expectedType),
      TyTy::WithLocation(capacityType, enuItem.getLocation()),
      enuItem.getLocation(), tcx);

  std::optional<CanonicalPath> path =
      tcx->lookupCanonicalPath(enuItem.getNodeId());
  assert(path.has_value());

  TypeIdentity ident(*path, enuItem.getLocation());

  return new TyTy::VariantDef(enuItem.getNodeId(), name, ident,
                              enuItem.getExpression().get());
}

// TyTy::BaseType
//   *TypeResolver::checkImplementationFunction(ast::TraitImpl *parent,
//   ast::Function *, TyTy::BaseType *self,
//   std::vector<TyTy::Substitutionparammapping> substitutions) {
//   assert(false);
//   }

} // namespace rust_compiler::sema::type_checking
