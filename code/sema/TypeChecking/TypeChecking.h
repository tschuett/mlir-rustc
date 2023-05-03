#pragma once

#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/AssignmentExpression.h"
#include "AST/AssociatedItem.h"
#include "AST/ClosureExpression.h"
#include "AST/ComparisonExpression.h"
#include "AST/Crate.h"
#include "AST/Expression.h"
#include "AST/ExpressionStatement.h"
#include "AST/ExternalItem.h"
#include "AST/Function.h"
#include "AST/GenericArgs.h"
#include "AST/GenericParams.h"
#include "AST/IfExpression.h"
#include "AST/IfLetExpression.h"
#include "AST/Implementation.h"
#include "AST/Item.h"
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
#include "AST/ReturnExpression.h"
#include "AST/StructStruct.h"
#include "AST/TupleStruct.h"
#include "AST/TypeParam.h"
#include "AST/Types/ArrayType.h"
#include "AST/Types/NeverType.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypePath.h"
#include "AST/WhereClause.h"
#include "Basic/Ids.h"
#include "TyCtx/NodeIdentity.h"
#include "TyCtx/TraitReference.h"
#include "TyCtx/TyCtx.h"
#include "TyCtx/TyTy.h"

// #include "../Resolver/Resolver.h"

#include <map>
#include <memory>
#include <vector>

namespace rust_compiler::sema::resolver {
class Resolver;
}

namespace rust_compiler::sema::type_checking {

using namespace rust_compiler::ast;
using namespace rust_compiler::tyctx;

class TypeCheckContextItem {
public:
  TypeCheckContextItem(ast::Function *f) : fun(f) {}

  ast::Function *getFunction() const { return fun; }
  TyTy::FunctionType *getContextType();

private:
  ast::Function *fun;
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

  // needed by substitutins.cpp in TyCtx
  TyTy::BaseType *checkType(std::shared_ptr<ast::types::TypeExpression>);

private:
  void checkVisItem(std::shared_ptr<ast::VisItem> v);
  void checkMacroItem(std::shared_ptr<ast::MacroItem> v);
  void checkFunction(std::shared_ptr<ast::Function> f);
  void checkStruct(ast::Struct *s);
  void checkStructStruct(ast::StructStruct *s);
  void checkTupleStruct(ast::TupleStruct *s);
  void checkWhereClause(const ast::WhereClause &);
  TyTy::BaseType *checkExpression(std::shared_ptr<ast::Expression>);
  TyTy::BaseType *
      checkExpressionWithBlock(std::shared_ptr<ast::ExpressionWithBlock>);
  TyTy::BaseType *
      checkExpressionWithoutBlock(std::shared_ptr<ast::ExpressionWithoutBlock>);
  TyTy::BaseType *checkBlockExpression(std::shared_ptr<ast::BlockExpression>);
  TyTy::BaseType *checkLiteral(std::shared_ptr<ast::LiteralExpression>);
  TyTy::BaseType *
      checkOperatorExpression(std::shared_ptr<ast::OperatorExpression>);
  TyTy::BaseType *checkArithmeticOrLogicalExpression(
      std::shared_ptr<ast::ArithmeticOrLogicalExpression>);
  TyTy::BaseType *checkReturnExpression(std::shared_ptr<ast::ReturnExpression>);
  TyTy::BaseType *checkMatchExpression(std::shared_ptr<ast::MatchExpression>);
  void checkGenericParams(const ast::GenericParams &);
  TyTy::BaseType *
      checkClosureExpression(std::shared_ptr<ast::ClosureExpression>);
  TyTy::BaseType *checkStatement(std::shared_ptr<ast::Statement>);
  TyTy::BaseType *checkPathExpression(std::shared_ptr<ast::PathExpression>);
  TyTy::BaseType *checkPathInExpression(std::shared_ptr<ast::PathInExpression>);
  TyTy::BaseType *checkLetStatement(std::shared_ptr<ast::LetStatement>);
  TyTy::BaseType *checkNeverType(std::shared_ptr<ast::types::NeverType>);
  TyTy::BaseType *
      checkExpressionStatement(std::shared_ptr<ast::ExpressionStatement>);
  TyTy::BaseType *checkIfExpression(std::shared_ptr<ast::IfExpression>);
  TyTy::BaseType *checkIfLetExpression(std::shared_ptr<ast::IfLetExpression>);
  TyTy::BaseType *
      checkAssignmentExpression(std::shared_ptr<ast::AssignmentExpression>);

  bool validateArithmeticType(ast::ArithmeticOrLogicalExpressionKind,
                              TyTy::BaseType *t);

  TyTy::BaseType *checkPattern(std::shared_ptr<ast::patterns::PatternNoTopAlt>,
                               TyTy::BaseType *);
  TyTy::BaseType *
  checkPatternWithoutRange(std::shared_ptr<ast::patterns::PatternWithoutRange>,
                           TyTy::BaseType *);
  TyTy::BaseType *
  checkRangePattern(std::shared_ptr<ast::patterns::RangePattern>,
                    TyTy::BaseType *);
  TyTy::BaseType *checkPathPattern(std::shared_ptr<ast::patterns::PathPattern>,
                                   TyTy::BaseType *);

  TyTy::BaseType *checkTypeNoBounds(std::shared_ptr<ast::types::TypeNoBounds>);
  TyTy::BaseType *checkTypePath(std::shared_ptr<ast::types::TypePath>);
  TyTy::BaseType *
      checkComparisonExpression(std::shared_ptr<ast::ComparisonExpression>);
  TyTy::BaseType *checkArrayType(std::shared_ptr<ast::types::ArrayType>);

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

  std::optional<TyTy::BaseType *> queryType(basic::NodeId id);

  bool
  resolveOperatorOverload(ArithmeticOrLogicalExpressionKind,
                          std::shared_ptr<ast::ArithmeticOrLogicalExpression>,
                          TyTy::BaseType *, TyTy::BaseType *);

  bool queryInProgress(basic::NodeId);
  void insertQuery(basic::NodeId);
  void queryCompleted(basic::NodeId);

  TyTy::BaseType *peekReturnType();
  void pushReturnType(TypeCheckContextItem item, TyTy::BaseType *returnRype);
  void popReturnType();

  TypeCheckContextItem &peekContext();

  std::vector<std::pair<TypeCheckContextItem, TyTy::BaseType *>>
      returnTypeStack;

  std::set<basic::NodeId> queriesInProgress;

  // data
  tyctx::TyCtx *tcx;
  resolver::Resolver *resolver;

  TyTy::BaseType *applyGenericArgs(TyTy::BaseType *, Location,
                                   const GenericArgs &);
  TyTy::BaseType *applyGenericArgsToADT(TyTy::ADTType *, Location,
                                        const GenericArgs &);
  //  TyTy::BaseType *
  //  applySubstitutionMappings(TyTy::BaseType *,
  //                            const TyTy::SubstitutionArgumentMappings &);

  TyTy::TypeBoundPredicate
      getPredicateFromBound(std::shared_ptr<ast::types::TypeExpression>);
  TyTy::ParamType *checkTypeParam(const TypeParam &);

  TraitReference *resolveTraitPath(std::shared_ptr<ast::types::TypePath>);

  //  TyTy::SubstitutionArgumentMappings
  //  getUsesSubstitutionArguments(TyTy::BaseType *);

  bool checkGenericParamsAndArgs(const TyTy::BaseType *, const ast::GenericArgs &);

  bool checkGenericParamsAndArgs(const ast::GenericParams &,
                                 const ast::GenericArgs &);
};

} // namespace rust_compiler::sema::type_checking
