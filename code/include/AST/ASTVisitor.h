#pragma once

#include "AST/Abi.h"
#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/ArrayExpression.h"
#include "AST/AsClause.h"
#include "AST/AssignmentExpression.h"
#include "AST/AssociatedItem.h"
#include "AST/AsyncBlockExpression.h"
#include "AST/Attr.h"
#include "AST/AttrInput.h"
#include "AST/BlockExpression.h"
#include "AST/BorrowExpression.h"
#include "AST/BreakExpression.h"
#include "AST/CallExpression.h"
#include "AST/ClosureExpression.h"
#include "AST/ComparisonExpression.h"
#include "AST/CompoundAssignmentExpression.h"
#include "AST/ConstParam.h"
#include "AST/ConstantItem.h"
#include "AST/ContinueExpression.h"
#include "AST/Crate.h"
#include "AST/CrateRef.h"
#include "AST/DereferenceExpression.h"
#include "AST/Enumeration.h"
#include "AST/ErrorPropagationExpression.h"
#include "AST/Expression.h"
#include "AST/ExpressionStatement.h"
#include "AST/ExternBlock.h"
#include "AST/ExternCrate.h"
#include "AST/ExternalItem.h"
#include "AST/FieldExpression.h"
#include "AST/Function.h"
#include "AST/FunctionParam.h"
#include "AST/FunctionParamPattern.h"
#include "AST/FunctionParameters.h"
#include "AST/FunctionQualifiers.h"
#include "AST/FunctionReturnType.h"
#include "AST/GenericParam.h"
#include "AST/GenericParams.h"
#include "AST/GroupedExpression.h"
#include "AST/IfExpression.h"
#include "AST/IfLetExpression.h"
#include "AST/Implementation.h"
#include "AST/IndexEpression.h"
#include "AST/InfiniteLoopExpression.h"
#include "AST/InherentImpl.h"
#include "AST/InnerAttribute.h"
#include "AST/IteratorLoopExpression.h"
#include "AST/LabelBlockExpression.h"
#include "AST/LazyBooleanExpression.h"
#include "AST/LetStatement.h"
#include "AST/LifetimeParam.h"
#include "AST/LiteralExpression.h"
#include "AST/LoopExpression.h"
#include "AST/MacroInvocationSemi.h"
#include "AST/MacroItem.h"
#include "AST/MatchExpression.h"
#include "AST/MethodCallExpression.h"
#include "AST/Module.h"
#include "AST/NegationExpression.h"
#include "AST/OperatorExpression.h"
#include "AST/OuterAttribute.h"
#include "AST/PathExpression.h"
#include "AST/PredicateLoopExpression.h"
#include "AST/PredicatePatternLoopExpression.h"
#include "AST/RangeExpression.h"
#include "AST/ReturnExpression.h"
#include "AST/SelfParam.h"
#include "AST/ShorthandSelf.h"
#include "AST/StaticItem.h"
#include "AST/Struct.h"
#include "AST/StructExpression.h"
#include "AST/Trait.h"
#include "AST/TraitImpl.h"
#include "AST/TupleExpression.h"
#include "AST/TupleIndexingExpression.h"
#include "AST/TypeAlias.h"
#include "AST/TypeCastExpression.h"
#include "AST/TypeParam.h"
#include "AST/TypedSelf.h"
#include "AST/UnderScoreExpression.h"
#include "AST/Union.h"
#include "AST/UnsafeBlockExpression.h"
#include "AST/UseDeclaration.h"
#include "AST/VisItem.h"
#include "AST/MacroInvocation.h"

namespace rust_compiler::ast {

class ASTVisitor {
public:
  virtual ~ASTVisitor() = default;

  virtual void visit(Crate &crate) = 0;

  virtual void visit(Item &item) = 0;
  virtual void visit(VisItem &visItem) = 0;
  virtual void visit(MacroItem &macroItem) = 0;

  // Attributes
  virtual void visit(OuterAttribute &outer) = 0;
  virtual void visit(InnerAttribute &inner) = 0;
  virtual void visit(Attr &attr) = 0;
  virtual void visit(AttrInput &attrInput) = 0;
  virtual void visit(DelimTokenTree &delimTokenTree) = 0;

  // VisItem
  virtual void visit(Module &module) = 0;
  virtual void visit(ExternCrate &externCrate) = 0;
  virtual void visit(UseDeclaration &useDeclaration) = 0;
  virtual void visit(Function &function) = 0;
  virtual void visit(TypeAlias &typeAlias) = 0;
  virtual void visit(Struct &structItem) = 0;
  virtual void visit(Enumeration &enumItem) = 0;
  virtual void visit(Union &unionItem) = 0;
  virtual void visit(ConstantItem &constantItem) = 0;
  virtual void visit(StaticItem &staticItem) = 0;
  virtual void visit(Trait &trait) = 0;
  virtual void visit(Implementation &implementation) = 0;
  virtual void visit(ExternBlock &externBlock) = 0;
  virtual void visit(AssociatedItem &associatedItem) = 0;

  // Extern Crate
  virtual void visit(CrateRef &crateRef) = 0;
  virtual void visit(AsClause &asClause) = 0;

  // UseDeclaration
  virtual void visit(use_tree::UseTree &useTree) = 0;

  // Function
  virtual void visit(FunctionQualifiers &qualifiers) = 0;
  virtual void visit(Abi &abi) = 0;
  virtual void visit(FunctionParameters &parameters) = 0;
  virtual void visit(SelfParam &selfParam) = 0;
  virtual void visit(ShorthandSelf &shortHandSelf) = 0;
  virtual void visit(TypedSelf &typedSelf) = 0;
  virtual void visit(FunctionParam &functionParam) = 0;
  virtual void visit(FunctionParamPattern &functionParamPattern) = 0;
  virtual void visit(FunctionReturnType &returnType) = 0;

  // Struct
  virtual void visit(class StructStruct &structStruct) = 0;
  virtual void visit(class TupleStruct &tupleStruct) = 0;
  virtual void visit(StructFields &structFields) = 0;
  virtual void visit(StructField &structField) = 0;
  virtual void visit(TupleFields &tupleFields) = 0;
  virtual void visit(TupleField &tupleField) = 0;

  // Enumeration
  virtual void visit(EnumItems &enumItems) = 0;
  virtual void visit(EnumItem &enumItem) = 0;
  virtual void visit(EnumItemTuple &enumItemTuple) = 0;
  virtual void visit(EnumItemStruct &enumItemStruct) = 0;
  virtual void visit(EnumItemDiscriminant &enumItemDiscriminant) = 0;

  // Implementation
  virtual void visit(InherentImpl &inherentImpl) = 0;
  virtual void visit(TraitImpl &traitImpl) = 0;

  // ExternBlock
  virtual void visit(ExternalItem &externalItem) = 0;

  // GenericParams
  virtual void visit(GenericParam &genericParam) = 0;
  virtual void visit(LifetimeParam &lifetimeParam) = 0;
  virtual void visit(TypeParam &typeParam) = 0;
  virtual void visit(ConstParam &constParam) = 0;

  // Statements
  virtual void visit(LetStatement &letStatement) = 0;
  virtual void visit(ExpressionStatement &expressionStatement) = 0;
  virtual void visit(MacroInvocationSemi &macroInvocationSemi) = 0;

  // Expressions without Block
  virtual void visit(LiteralExpression &literalExpression) = 0;
  virtual void visit(PathExpression &pathExpression) = 0;
  virtual void visit(GroupedExpression &groupedExpression) = 0;
  virtual void visit(ArrayExpression &arrayExpression) = 0;
  virtual void visit(IndexExpression &indexExpression) = 0;
  virtual void visit(TupleExpression &indexExpression) = 0;
  virtual void visit(TupleIndexingExpression &indexExpression) = 0;
  virtual void visit(StructExpression &indexExpression) = 0;
  virtual void visit(CallExpression &indexExpression) = 0;
  virtual void visit(MethodCallExpression &indexExpression) = 0;
  virtual void visit(FieldExpression &indexExpression) = 0;
  virtual void visit(ClosureExpression &indexExpression) = 0;
  virtual void visit(AsyncBlockExpression &indexExpression) = 0;
  virtual void visit(ContinueExpression &indexExpression) = 0;
  virtual void visit(BreakExpression &indexExpression) = 0;
  virtual void visit(RangeExpression &indexExpression) = 0;
  virtual void visit(ReturnExpression &indexExpression) = 0;
  virtual void visit(UnderScoreExpression &indexExpression) = 0;
  virtual void visit(MacroInvocation &indexExpression) = 0;

  // Expressions with Block
  virtual void visit(BlockExpression &blockExpression) = 0;
  virtual void visit(UnsafeBlockExpression &unsafeBlockExpression) = 0;
  virtual void visit(IfExpression &ifExpression) = 0;
  virtual void visit(IfLetExpression &ifLetExpression) = 0;
  virtual void visit(MatchExpression &matchExpression) = 0;

  // Operator Expressions
  virtual void visit(BorrowExpression &borrowExpression) = 0;
  virtual void visit(DereferenceExpression &borrowExpression) = 0;
  virtual void visit(ErrorPropagationExpression &borrowExpression) = 0;
  virtual void visit(NegationExpression &borrowExpression) = 0;
  virtual void visit(ArithmeticOrLogicalExpression &borrowExpression) = 0;
  virtual void visit(ComparisonExpression &borrowExpression) = 0;
  virtual void visit(LazyBooleanExpression &borrowExpression) = 0;
  virtual void visit(TypeCastExpression &borrowExpression) = 0;
  virtual void visit(AssignmentExpression &borrowExpression) = 0;
  virtual void visit(CompoundAssignmentExpression &borrowExpression) = 0;

  // LoopExpression
  virtual void visit(InfiniteLoopExpression &loopExpression) = 0;
  virtual void visit(PredicateLoopExpression &loopExpression) = 0;
  virtual void visit(PredicatePatternLoopExpression &loopExpression) = 0;
  virtual void visit(IteratorLoopExpression &loopExpression) = 0;
  virtual void visit(LabelBlockExpression &loopExpression) = 0;

  // Match Expressions
  virtual void visit(Scrutinee &loopExpression) = 0;
  virtual void visit(MatchArms &loopExpression) = 0;
  virtual void visit(MatchArm &loopExpression) = 0;
  virtual void visit(MatchArmGuard &loopExpression) = 0;

  // Path expression
};

} // namespace rust_compiler::ast

// FIXME: patterns?
