#pragma once

#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/ArrayExpression.h"
#include "AST/AssignmentExpression.h"
#include "AST/AssociatedItem.h"
#include "AST/BorrowExpression.h"
#include "AST/CallExpression.h"
#include "AST/ClosureExpression.h"
#include "AST/ComparisonExpression.h"
#include "AST/CompoundAssignmentExpression.h"
#include "AST/Crate.h"
#include "AST/EnumItemDiscriminant.h"
#include "AST/EnumItemStruct.h"
#include "AST/EnumItemTuple.h"
#include "AST/Expression.h"
#include "AST/ExpressionStatement.h"
#include "AST/ExternalItem.h"
#include "AST/FieldExpression.h"
#include "AST/Function.h"
#include "AST/GenericArgs.h"
#include "AST/GenericParams.h"
#include "AST/IfExpression.h"
#include "AST/IfLetExpression.h"
#include "AST/Implementation.h"
#include "AST/IndexEpression.h"
#include "AST/InherentImpl.h"
#include "AST/Item.h"
#include "AST/ItemDeclaration.h"
#include "AST/IteratorLoopExpression.h"
#include "AST/LetStatement.h"
#include "AST/LiteralExpression.h"
#include "AST/MacroItem.h"
#include "AST/MatchExpression.h"
#include "AST/OperatorExpression.h"
#include "AST/PathExpression.h"
#include "AST/PathInExpression.h"
#include "AST/Patterns/PathPattern.h"
#include "AST/Patterns/PatternWithoutRange.h"
#include "AST/Patterns/RangePattern.h"
#include "AST/Patterns/StructPattern.h"
#include "AST/Patterns/TupleStructPattern.h"
#include "AST/RangeExpression.h"
#include "AST/ReturnExpression.h"
#include "AST/StructExprStruct.h"
#include "AST/StructExpression.h"
#include "AST/StructStruct.h"
#include "AST/TraitImpl.h"
#include "AST/TupleExpression.h"
#include "AST/TupleIndexingExpression.h"
#include "AST/TupleStruct.h"
#include "AST/TypeAlias.h"
#include "AST/TypeCastExpression.h"
#include "AST/TypeParam.h"
#include "AST/Types/ArrayType.h"
#include "AST/Types/NeverType.h"
#include "AST/Types/RawPointerType.h"
#include "AST/Types/ReferenceType.h"
#include "AST/Types/SliceType.h"
#include "AST/Types/TraitObjectTypeOneBound.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypePath.h"
#include "AST/UnsafeBlockExpression.h"
#include "AST/WhereClause.h"
#include "Basic/Ids.h"
#include "PathProbing.h"
#include "Sema/Properties.h"
#include "TyCtx/NodeIdentity.h"
#include "TyCtx/Substitutions.h"
#include "TyCtx/TraitReference.h"
#include "TyCtx/TyCtx.h"
#include "TyCtx/TyTy.h"

// #include "../Resolver/Resolver.h"

#include <map>
#include <memory>
#include <stack>
#include <variant>
#include <vector>

namespace rust_compiler::ast {
class InherentImpl;
class Trait;
class MethodCallExpression;
} // namespace rust_compiler::ast

namespace rust_compiler::sema::resolver {
class Resolver;
}

namespace rust_compiler::sema::type_checking {

using namespace rust_compiler::ast;
using namespace rust_compiler::tyctx;

class TypeCheckContextItem {
  enum ItemKind { Function };

public:
  TypeCheckContextItem(ast::Function *f) : kind(ItemKind::Function), item(f) {}

  ast::Function *getFunction() const { return std::get<ast::Function *>(item); }
  TyTy::FunctionType *getContextType();

private:
  ItemKind getKind() const { return kind; }
  ItemKind kind;
  std::variant<ast::Function *> item;
};

/// https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_analysis/index.html
class TypeResolver {
public:
  TypeResolver(resolver::Resolver *);

  void checkCrate(std::shared_ptr<ast::Crate> crate);

  TyTy::BaseType *checkEnumerationPointer(ast::Enumeration *e);
  TyTy::BaseType *checkImplementationPointer(ast::Implementation *i);
  TyTy::BaseType *checkExternalItemPointer(ast::ExternalItem *e);
  TyTy::BaseType *checkItemPointer(ast::Item *e);
  TyTy::BaseType *checkAssociatedItemPointer(ast::AssociatedItem *,
                                             ast::Implementation *);

  // FIXME: needed by substitutins.cpp in TyCtx
  TyTy::BaseType *checkType(std::shared_ptr<ast::types::TypeExpression>);

  std::optional<TyTy::BaseType *> queryType(basic::NodeId id);

private:
  std::optional<TyTy::BaseType *>
  checkFunctionTraitCall(CallExpression *, TyTy::BaseType *functionType);
  std::optional<tyctx::TyTy::FunctionTrait>
  checkPossibleFunctionTraitCallMethodName(TyTy::BaseType &receiver,
                                           TyTy::TypeBoundPredicate *);

  void checkVisItem(std::shared_ptr<ast::VisItem> v);
  void checkMacroItem(std::shared_ptr<ast::MacroItem> v);
  void checkFunction(std::shared_ptr<ast::Function> f);
  void checkStruct(ast::Struct *s);
  void checkConstantItem(ast::ConstantItem *);
  void checkTypeAlias(ast::TypeAlias *);
  void checkEnumeration(ast::Enumeration *);
  TyTy::VariantDef *checkEnumItem(EnumItem *, int64_t discriminant);
  TyTy::VariantDef *checkEnumItemTuple(const EnumItemTuple &enuItem,
                                       const Identifier &,
                                       int64_t discriminant);
  TyTy::VariantDef *checkEnumItemStruct(const EnumItemStruct &enuItem,
                                        const Identifier &,
                                        int64_t discriminant);
  TyTy::VariantDef *
  checkEnumItemDiscriminant(const EnumItemDiscriminant &enuItem,
                            const Identifier &, int64_t discriminant);

  TyTy::BaseType *checkItemDeclaration(ast::ItemDeclaration *item);
  TyTy::BaseType *checkTrait(ast::Trait *s);
  void checkStructStruct(ast::StructStruct *s);
  void checkTupleStruct(ast::TupleStruct *s);
  void checkWhereClause(const ast::WhereClause &);
  TyTy::BaseType *checkExpression(std::shared_ptr<ast::Expression>);
  TyTy::BaseType *
      checkExpressionWithBlock(std::shared_ptr<ast::ExpressionWithBlock>);
  TyTy::BaseType *
      checkExpressionWithoutBlock(std::shared_ptr<ast::ExpressionWithoutBlock>);
  TyTy::BaseType *checkBlockExpression(std::shared_ptr<ast::BlockExpression>);
  TyTy::BaseType *
      checkUnsafeBlockExpression(std::shared_ptr<ast::UnsafeBlockExpression>);
  TyTy::BaseType *checkLoopExpression(std::shared_ptr<ast::LoopExpression>);
  TyTy::BaseType *checkIteratorLoopExpression(ast::IteratorLoopExpression *);
  TyTy::BaseType *checkStructExpression(ast::StructExpression *);
  TyTy::BaseType *checkStructExprStruct(ast::StructExprStruct *);

  TyTy::BaseType *checkLiteral(std::shared_ptr<ast::LiteralExpression>);
  TyTy::BaseType *
      checkOperatorExpression(std::shared_ptr<ast::OperatorExpression>);
  TyTy::BaseType *checkArrayExpression(std::shared_ptr<ast::ArrayExpression>);

  TyTy::BaseType *checkArithmeticOrLogicalExpression(
      std::shared_ptr<ast::ArithmeticOrLogicalExpression>);
  TyTy::BaseType *checkBorrowExpression(std::shared_ptr<ast::BorrowExpression>);
  TyTy::BaseType *checkIndexExpression(std::shared_ptr<ast::IndexExpression>);
  TyTy::BaseType *checkTupleIndexingExpression(ast::TupleIndexingExpression *);

  TyTy::BaseType *
      checkTypeCastExpression(std::shared_ptr<ast::TypeCastExpression>);

  TyTy::BaseType *checkReturnExpression(std::shared_ptr<ast::ReturnExpression>);
  TyTy::BaseType *checkMatchExpression(std::shared_ptr<ast::MatchExpression>);
  void checkGenericParams(const ast::GenericParams &,
                          std::vector<TyTy::SubstitutionParamMapping> &);
  TyTy::BaseType *
      checkClosureExpression(std::shared_ptr<ast::ClosureExpression>);
  TyTy::BaseType *checkStatement(std::shared_ptr<ast::Statement>);
  TyTy::BaseType *checkPathExpression(std::shared_ptr<ast::PathExpression>);
  TyTy::BaseType *checkPathInExpression(std::shared_ptr<ast::PathInExpression>);
  TyTy::BaseType *checkLetStatement(std::shared_ptr<ast::LetStatement>);
  TyTy::BaseType *checkNeverType(std::shared_ptr<ast::types::NeverType>);
  TyTy::BaseType *
      checkReferenceType(std::shared_ptr<ast::types::ReferenceType>);
  TyTy::BaseType *checkFieldExpression(ast::FieldExpression *);

  TyTy::BaseType *
      checkExpressionStatement(std::shared_ptr<ast::ExpressionStatement>);
  TyTy::BaseType *checkIfExpression(std::shared_ptr<ast::IfExpression>);
  TyTy::BaseType *checkIfLetExpression(std::shared_ptr<ast::IfLetExpression>);
  TyTy::BaseType *
      checkAssignmentExpression(std::shared_ptr<ast::AssignmentExpression>);
  TyTy::BaseType *checkCompoundAssignmentExpression(
      std::shared_ptr<ast::CompoundAssignmentExpression>);

  bool validateArithmeticType(ast::ArithmeticOrLogicalExpressionKind,
                              TyTy::BaseType *t);
  bool validateArithmeticType(ast::CompoundAssignmentExpressionKind,
                              TyTy::BaseType *t);

  TyTy::BaseType *checkPattern(std::shared_ptr<ast::patterns::PatternNoTopAlt>,
                               TyTy::BaseType *);
  TyTy::BaseType *checkPattern(std::shared_ptr<ast::patterns::Pattern>,
                               TyTy::BaseType *);
  TyTy::BaseType *
  checkPatternWithoutRange(std::shared_ptr<ast::patterns::PatternWithoutRange>,
                           TyTy::BaseType *);
  TyTy::BaseType *
  checkRangePattern(std::shared_ptr<ast::patterns::RangePattern>,
                    TyTy::BaseType *);
  TyTy::BaseType *checkPathPattern(std::shared_ptr<ast::patterns::PathPattern>,
                                   TyTy::BaseType *);
  TyTy::BaseType *
  checkTupleStructPattern(std::shared_ptr<ast::patterns::TupleStructPattern>,
                          TyTy::BaseType *);
  TyTy::BaseType *
  checkStructPattern(std::shared_ptr<ast::patterns::StructPattern>,
                     TyTy::BaseType *);

  TyTy::BaseType *checkTypeNoBounds(std::shared_ptr<ast::types::TypeNoBounds>);
  TyTy::BaseType *checkTypePath(std::shared_ptr<ast::types::TypePath>);
  TyTy::BaseType *checkTypeTraitObjectTypeOneBound(
      std::shared_ptr<ast::types::TraitObjectTypeOneBound>);

  TyTy::BaseType *
      checkComparisonExpression(std::shared_ptr<ast::ComparisonExpression>);
  TyTy::BaseType *checkTupleExpression(std::shared_ptr<ast::TupleExpression>);
  TyTy::BaseType *checkArrayType(std::shared_ptr<ast::types::ArrayType>);
  TyTy::BaseType *checkCallExpression(ast::CallExpression *);
  TyTy::BaseType *checkCallExpression(TyTy::BaseType *functionType,
                                      ast::CallExpression *,
                                      TyTy::VariantDef &);
  TyTy::BaseType *checkCallExpressionFn(TyTy::BaseType *functionType,
                                        ast::CallExpression *,
                                        TyTy::VariantDef &);
  TyTy::BaseType *checkCallExpressionADT(TyTy::BaseType *functionType,
                                         ast::CallExpression *,
                                         TyTy::VariantDef &);
  TyTy::BaseType *checkMethodCallExpression(ast::MethodCallExpression *);

  TyTy::BaseType *
  resolveRootPathType(std::shared_ptr<ast::types::TypePath> path,
                      size_t *offset, basic::NodeId *resolvedNodeId);
  TyTy::BaseType *resolveSegmentsType(basic::NodeId resolvedNodeId,
                                      basic::NodeId pathNodeId,
                                      std::shared_ptr<ast::types::TypePath> tp,
                                      size_t offset, TyTy::BaseType *pathType);
  TyTy::BaseType *
  resolveRootPathExpression(std::shared_ptr<ast::PathInExpression> path,
                            size_t *offset, basic::NodeId *resolvedNodeId);
  TyTy::BaseType *resolveSegmentsExpression(basic::NodeId rootResolvedIt,
                                            std::span<PathExprSegment> segment,
                                            size_t offset,
                                            TyTy::BaseType *typeSegment,
                                            tyctx::NodeIdentity, Location);

  std::set<MethodCandidate> resolveMethodProbe(TyTy::BaseType *receiver,
                                               TyTy::FunctionTrait);
  TyTy::BaseType *checkMethodCallExpression(TyTy::FunctionType *, NodeIdentity,
                                            std::vector<TyTy::Argument> &args,
                                            Location call, Location receiver,
                                            TyTy::BaseType *adjustedSelf);
  void checkImplementation(ast::Implementation *);
  void checkInherentImpl(ast::InherentImpl *);
  void checkTraitImpl(ast::TraitImpl *);
  TyTy::ParamType *checkGenericParam(const ast::GenericParam &);
  TyTy::ParamType *checkGenericParamTypeParam(const ast::TypeParam &);
  TyTy::BaseType *
      checkRawPointerType(std::shared_ptr<ast::types::RawPointerType>);
  TyTy::BaseType *checkSliceType(std::shared_ptr<ast::types::SliceType>);
  TyTy::BaseType *checkIntoIteratorElementType(ast::Expression *);
  TyTy::BaseType *checkIntoIteratorElementType(ast::ExpressionWithoutBlock *);
  TyTy::BaseType *checkIntoIteratorElementType(ast::PathExpression *);
  TyTy::BaseType *checkIntoIteratorElementType(TyTy::BaseType *);
  TyTy::BaseType *checkIntoIteratorElementType(ast::RangeExpression *);

  TyTy::BaseType *resolveImplBlockSelfWithInference(
      ast::Implementation *impl, Location loc,
      TyTy::SubstitutionArgumentMappings *inferArguments);

  bool
  resolveOperatorOverload(ArithmeticOrLogicalExpressionKind,
                          std::shared_ptr<ast::ArithmeticOrLogicalExpression>,
                          TyTy::BaseType *, TyTy::BaseType *);

  std::optional<TyTy::BaseType *>
  resolveOperatorOverloadIndexTrait(ast::IndexExpression *, TyTy::BaseType *,
                                    TyTy::BaseType *);

  bool queryInProgress(basic::NodeId);
  void insertQuery(basic::NodeId);
  void queryCompleted(basic::NodeId);

  TyTy::BaseType *peekReturnType();
  void pushReturnType(TypeCheckContextItem item, TyTy::BaseType *returnRype);
  void popReturnType();
  void pushSmallSelf(TyTy::BaseType *self) { smallSelfStack.push(self); }
  void popSmallSelf() { smallSelfStack.pop(); }
  bool hasSmallSelf() { return smallSelfStack.size() > 0; }
  TyTy::BaseType *getSmallSelf() { return smallSelfStack.top(); }

  TypeCheckContextItem &peekContext();

  std::vector<std::pair<TypeCheckContextItem, TyTy::BaseType *>>
      returnTypeStack;

  std::stack<TyTy::BaseType *> smallSelfStack;

  bool haveFunctionContext() const { return not returnTypeStack.empty(); }

  std::set<basic::NodeId> queriesInProgress;

  // data
  tyctx::TyCtx *tcx;
  resolver::Resolver *resolver;

  TyTy::BaseType *applyGenericArgs(TyTy::BaseType *, Location,
                                   const GenericArgs &);
  TyTy::BaseType *applyGenericArgsToADT(TyTy::ADTType *, Location,
                                        const GenericArgs &);
  TyTy::BaseType *
  applySubstitutionMappings(TyTy::BaseType *,
                            const TyTy::SubstitutionArgumentMappings &);

  TyTy::TypeBoundPredicate
  getPredicateFromBound(std::shared_ptr<ast::types::TypeExpression>,
                        ast::types::TypeExpression *);
  TyTy::ParamType *checkTypeParam(const TypeParam &);

  TyTy::TraitReference *resolveTraitPath(std::shared_ptr<ast::types::TypePath>);

  TyTy::SubstitutionArgumentMappings
  getUsedSubstitutionArguments(TyTy::BaseType *);

  bool checkGenericParamsAndArgs(const TyTy::BaseType *,
                                 const ast::GenericArgs &);

  bool checkGenericParamsAndArgs(const ast::GenericParams &,
                                 const ast::GenericArgs &);
  TyTy::TraitReference *resolveTrait(ast::Trait *trait);

  TyTy::TraitItemReference resolveAssociatedItemInTraitToRef(
      AssociatedItem &, TyTy::BaseType *,
      const std::vector<TyTy::SubstitutionParamMapping> &);

  TyTy::BaseType *inferClosureParam(patterns::PatternNoTopAlt *);

  void resolveFunctionItemInTrait(std::shared_ptr<ast::Item>,
                                  TyTy::BaseType *type);

  std::optional<Trait *>
  resolvePathToTrait(std::shared_ptr<ast::types::TypePath> path);

  std::optional<TyTy::BaseType *> resolveOperatorOverload(PropertyKind,
                                                          Expression *,
                                                          TyTy::BaseType *lhs,
                                                          TyTy::BaseType *rhs);

  std::set<MethodCandidate>
  probeMethodResolver(TyTy::BaseType *receiver,
                      const ast::PathIdentSegment &segmentName,
                      bool autoDeref = false);

  std::optional<std::vector<TyTy::SubstitutionParamMapping>>
  resolveInherentImplSubstitutions(InherentImpl *impl);
  std::optional<std::vector<TyTy::SubstitutionParamMapping>>
  resolveTraitImplSubstitutions(TraitImpl *impl);

  bool checkForUnconstrained(
      const std::vector<TyTy::SubstitutionParamMapping> &paramsToConstrain,
      const TyTy::SubstitutionArgumentMappings &constraintA,
      const TyTy::SubstitutionArgumentMappings &constraintB,
      const TyTy::BaseType *reference);

  TyTy::BaseType *resolveInherentImplSelf(InherentImpl *);
  TyTy::BaseType *resolveTraitImplSelf(TraitImpl *);

  void validateTraitImplBlock(
      TraitImpl *, TyTy::BaseType *self,
      std::vector<TyTy::SubstitutionParamMapping> &substitutions);

  void validateInherentImplBlock(
      InherentImpl *, TyTy::BaseType *self,
      std::vector<TyTy::SubstitutionParamMapping> &substitutions);

  void
  checkTraitImplItem(ast::TraitImpl *impl, AssociatedItem &asso,
                     TyTy::BaseType *self,
                     std::vector<TyTy::SubstitutionParamMapping> substitutions);
  void checkInherentImplItem(
      ast::InherentImpl *impl, AssociatedItem &asso, TyTy::BaseType *self,
      std::vector<TyTy::SubstitutionParamMapping> substitutions);

  TyTy::BaseType *checkImplementationFunction(
      ast::TraitImpl *parent, ast::Function *, TyTy::BaseType *self,
      std::vector<TyTy::SubstitutionParamMapping> substitutions);

  TyTy::BaseType *checkImplementationFunction(
      ast::InherentImpl *parent, ast::Function *, TyTy::BaseType *self,
      std::vector<TyTy::SubstitutionParamMapping> substitutions);

  TyTy::BaseType *resolveImplBlockSelf(const AssociatedImplTrait &);
  TyTy::BaseType *resolveImplBlockSelf(const Implementation *);

  std::vector<TyTy::SubstitutionParamMapping>
  resolveImplBlockSubstitutions(ast::Implementation *impl, bool &failedFlag);
  std::vector<TyTy::SubstitutionParamMapping>
  resolveImplBlockSubstitutions(ast::InherentImpl *impl, bool &failedFlag);
  std::vector<TyTy::SubstitutionParamMapping>
  resolveImplBlockSubstitutions(ast::TraitImpl *impl, bool &failedFlag);
};

} // namespace rust_compiler::sema::type_checking
