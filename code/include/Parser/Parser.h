#pragma once

#include "ADT/Result.h"
#include "AST/Abi.h"
#include "AST/ArrayElements.h"
#include "AST/AssociatedItem.h"
#include "AST/AttrInput.h"
#include "AST/CallParams.h"
#include "AST/ClosureParameters.h"
#include "AST/ConstParam.h"
#include "AST/Crate.h"
#include "AST/EnumItem.h"
#include "AST/EnumItemDiscriminant.h"
#include "AST/EnumItems.h"
#include "AST/ExternalItem.h"
#include "AST/Function.h"
#include "AST/FunctionParam.h"
#include "AST/FunctionParamPattern.h"
#include "AST/FunctionParameters.h"
#include "AST/FunctionReturnType.h"
#include "AST/GenericArgsBinding.h"
#include "AST/GenericArgsConst.h"
#include "AST/GenericParam.h"
#include "AST/Lifetime.h"
#include "AST/LifetimeBounds.h"
#include "AST/LifetimeParam.h"
#include "AST/LifetimeWhereClauseItem.h"
#include "AST/LoopExpression.h"
#include "AST/LoopLabel.h"
#include "AST/MacroRulesDef.h"
#include "AST/MatchArm.h"
#include "AST/MatchArmGuard.h"
#include "AST/MatchArms.h"
#include "AST/OuterAttribute.h"
#include "AST/PathExprSegment.h"
#include "AST/PathExpression.h"
#include "AST/Patterns/Pattern.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Patterns/SlicePatternItems.h"
#include "AST/Patterns/StructPattern.h"
#include "AST/Patterns/StructPatternElements.h"
#include "AST/Patterns/TuplePatternItems.h"
#include "AST/Patterns/TupleStructItems.h"
#include "AST/Scrutinee.h"
#include "AST/SelfParam.h"
#include "AST/Statements.h"
#include "AST/StructExprField.h"
#include "AST/StructExprFields.h"
#include "AST/TupleElements.h"
#include "AST/TypeAlias.h"
#include "AST/TypeBoundWhereClauseItem.h"
#include "AST/TypeParam.h"
#include "AST/Types/BareFunctionType.h"
#include "AST/Types/QualifiedPathType.h"
#include "AST/Types/TraitBound.h"
#include "AST/Types/TypeParamBound.h"
#include "AST/Types/TypeParamBounds.h"
#include "AST/Types/TypePathFnInputs.h"
#include "AST/UseTree.h"
#include "AST/VisItem.h"
#include "AST/Visiblity.h"
#include "AST/WhereClauseItem.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Lexer/TokenStream.h"
#include "Location.h"
#include "Parser/ErrorStack.h"
#include "Parser/Precedence.h"
#include "Parser/Restrictions.h"
#include "Parser/TokenPointer.h"

#include <span>
#include <stack>
#include <string>
#include <string_view>

/// https://doc.rust-lang.org/nightly/nightly-rustc/rustc_parse/parser/struct.Parser.html#method.new
namespace rust_compiler::parser {

class CheckPoint {
  size_t offset = 0;

public:
  CheckPoint(size_t offset) : offset(offset) {}
  size_t readOffset() const { return offset; }
};

enum PathKind { TypePath, SimplePath, Unknown };

class Parser {
  lexer::TokenStream ts;

  size_t offset = 0;

  rust_compiler::Location getLocation();

public:
  Parser(lexer::TokenStream &ts) : ts(ts){};

  adt::Result<ast::Visibility, std::string> parseVisibility();
  adt::Result<ast::use_tree::UseTree, std::string> parseUseTree();

  adt::Result<std::shared_ptr<ast::Item>, std::string> parseItem();
  adt::Result<std::shared_ptr<ast::Item>, std::string>
      parseMacroItem(std::span<ast::OuterAttribute>);

  adt::Result<std::shared_ptr<ast::Item>, std::string>
      parseVisItem(std::span<ast::OuterAttribute>);

  adt::Result<std::shared_ptr<ast::patterns::PatternNoTopAlt>, std::string>
  parseRangeOrIdentifierOrStructOrTupleStructOrMacroInvocationPattern();

  /// VisItems
  adt::Result<std::shared_ptr<ast::Item>, std::string>
  parseMod(std::optional<ast::Visibility> vis);
  adt::Result<std::shared_ptr<ast::Item>, std::string>
  parseUseDeclaration(std::optional<ast::Visibility> vis);
  adt::Result<std::shared_ptr<ast::Item>, std::string>
  parseTypeAlias(std::optional<ast::Visibility> vis);
  adt::Result<std::shared_ptr<ast::Item>, std::string>
  parseEnumeration(std::optional<ast::Visibility> vis);
  adt::Result<std::shared_ptr<ast::Item>, std::string>
  parseUnion(std::optional<ast::Visibility> vis);
  adt::Result<std::shared_ptr<ast::Item>, std::string>
  parseStaticItem(std::optional<ast::Visibility> vis);
  adt::Result<std::shared_ptr<ast::Item>, std::string>
  parseTrait(std::optional<ast::Visibility> vis);
  adt::Result<std::shared_ptr<ast::Item>, std::string>
  parseFunction(std::optional<ast::Visibility> vis);
  adt::Result<std::shared_ptr<ast::Item>, std::string>
  parseConstantItem(std::optional<ast::Visibility> vis);
  adt::Result<std::shared_ptr<ast::Item>, std::string>
  parseExternBlock(std::optional<ast::Visibility> vis);

  adt::Result<ast::ExternalItem, std::string> parseExternalItem();

  adt::Result<std::shared_ptr<ast::Item>, std::string>
  parseStruct(std::optional<ast::Visibility> vis);
  adt::Result<ast::StructFields, std::string> parseStructFields();
  adt::Result<ast::StructField, std::string> parseStructField();
  adt::Result<std::shared_ptr<ast::Item>, std::string>
  parseStructStruct(std::optional<ast::Visibility> vis);
  adt::Result<std::shared_ptr<ast::Item>, std::string>
  parseTupleStruct(std::optional<ast::Visibility> vis);

  adt::Result<std::shared_ptr<ast::Item>, std::string>
  parseImplementation(std::optional<ast::Visibility> vis);
  adt::Result<std::shared_ptr<ast::Item>, std::string>
  parseInherentImpl(std::optional<ast::Visibility> vis);
  adt::Result<std::shared_ptr<ast::Item>, std::string>
  parseTraitImpl(std::optional<ast::Visibility> vis);

  adt::Result<std::vector<ast::OuterAttribute>, std::string>
  parseOuterAttributes();
  adt::Result<std::vector<ast::InnerAttribute>, std::string>
  parseInnerAttributes();

  adt::Result<ast::OuterAttribute, std::string> parseOuterAttribute();
  adt::Result<ast::InnerAttribute, std::string> parseInnerAttribute();

  adt::Result<ast::ConstParam, std::string> parseConstParam();
  adt::Result<ast::LifetimeParam, std::string> parseLifetimeParam();
  adt::Result<ast::TypeParam, std::string> parseTypeParam();

  // Function
  adt::Result<ast::FunctionQualifiers, std::string> parseFunctionQualifiers();
  // llvm::Expected<ast::FunctionSignature> parseFunctionsignature();
  adt::Result<ast::FunctionParam, std::string> parseFunctionParam();
  adt::Result<ast::FunctionParameters, std::string> parseFunctionParameters();
  adt::Result<ast::FunctionParamPattern, std::string>
  parseFunctionParamPattern();
  adt::Result<ast::FunctionReturnType, std::string> parseFunctionReturnType();
  adt::Result<ast::SelfParam, std::string> parseSelfParam();

  adt::Result<ast::Statements, std::string> parseStatements();
  adt::Result<std::shared_ptr<ast::Statement>, std::string>
  parseStatement(Restrictions restriction);

  // Types
  adt::Result<std::shared_ptr<ast::types::TypeExpression>, std::string>
  parseTypeNoBounds();
  adt::Result<std::shared_ptr<ast::types::TypeExpression>, std::string>
  parseType();
  adt::Result<std::shared_ptr<ast::types::TypeExpression>, std::string>
  parseTypePath();

  adt::Result<ast::types::TypePathSegment, std::string> parseTypePathSegment();
  adt::Result<ast::PathIdentSegment, std::string> parsePathIdentSegment();

  adt::Result<ast::PathExprSegment, std::string> parsePathExprSegment();

  adt::Result<ast::types::TypePathFn, std::string> parseTypePathFn();
  adt::Result<ast::types::TypePathFnInputs, std::string>
  parseTypePathFnInputs();

  adt::Result<std::shared_ptr<ast::types::TypeParamBound>, std::string>
  parseTraitBound();
  adt::Result<std::shared_ptr<ast::types::TypeExpression>, std::string>
  parseBareFunctionType();
  adt::Result<ast::types::FunctionTypeQualifiers, std::string>
  parseFunctionTypeQualifiers();
  adt::Result<ast::types::BareFunctionReturnType, std::string>
  parseBareFunctionReturnType();

  adt::Result<std::shared_ptr<ast::types::TypeExpression>, std::string>
  parseTraitObjectTypeOneBound();
  adt::Result<std::shared_ptr<ast::types::TypeExpression>, std::string>
  parseImplTraitTypeOneBound();

  adt::Result<std::shared_ptr<ast::types::TypeExpression>, std::string>
  parseImplTraitType();

  adt::Result<std::shared_ptr<ast::types::TypeExpression>, std::string>
  parseRawPointerType();
  adt::Result<std::shared_ptr<ast::types::TypeExpression>, std::string>
  parseNeverType();
  adt::Result<std::shared_ptr<ast::types::TypeExpression>, std::string>
  parseReferenceType();
  adt::Result<std::shared_ptr<ast::types::TypeExpression>, std::string>
  parseInferredType();
  adt::Result<std::shared_ptr<ast::types::TypeExpression>, std::string>
  parseTraitObjectType();
  adt::Result<std::shared_ptr<ast::types::TypeExpression>, std::string>
  parseImplType();
  adt::Result<std::shared_ptr<ast::types::TypeExpression>, std::string>
  parseArrayOrSliceType();
  adt::Result<std::shared_ptr<ast::types::TypeExpression>, std::string>
  parseTupleOrParensType();
  adt::Result<std::shared_ptr<ast::types::TypeExpression>, std::string>
  parseTupleType();
  adt::Result<std::shared_ptr<ast::types::TypeExpression>, std::string>
  parseTupleOrParensTypeOrTypePathOrMacroInvocationOrTraitObjectTypeOrBareFunctionType();
  // only for types !!!

  adt::Result<std::shared_ptr<ast::types::TypeExpression>, std::string>
  parseQualifiedPathInType();
  adt::Result<ast::types::QualifiedPathType, std::string>
  parseQualifiedPathType();

  adt::Result<std::shared_ptr<ast::types::TypeExpression>, std::string>
  parseParenthesizedType();

  PathKind testTypePathOrSimplePath();

  adt::Result<std::shared_ptr<ast::Crate>, std::string>
  parseCrateModule(std::string_view crateName, basic::CrateNum crateNum);

  // Patterns
  adt::Result<std::shared_ptr<ast::patterns::Pattern>, std::string>
  parsePattern();
  adt::Result<std::shared_ptr<ast::patterns::PatternNoTopAlt>, std::string>
  parsePatternNoTopAlt();
  adt::Result<std::shared_ptr<ast::patterns::PatternNoTopAlt>, std::string>
  parseReferencePattern();
  adt::Result<std::shared_ptr<ast::patterns::PatternNoTopAlt>, std::string>
  parseRestPattern();
  adt::Result<std::shared_ptr<ast::patterns::PatternNoTopAlt>, std::string>
  parseWildCardPattern();
  adt::Result<std::shared_ptr<ast::patterns::PatternNoTopAlt>, std::string>
  parseTupleOrGroupedPattern();
  adt::Result<std::shared_ptr<ast::patterns::PatternNoTopAlt>, std::string>
  parseSlicePattern();
  adt::Result<std::shared_ptr<ast::patterns::PatternNoTopAlt>, std::string>
  parseLiteralPattern();
  adt::Result<ast::patterns::SlicePatternItems, std::string>
  parseSlicePatternItems();
  adt::Result<std::shared_ptr<ast::patterns::PatternNoTopAlt>, std::string>
  parsePatternWithoutRange();

  adt::Result<std::shared_ptr<ast::patterns::PatternNoTopAlt>, std::string>
  parseRangePattern();
  adt::Result<std::shared_ptr<ast::patterns::PatternNoTopAlt>, std::string>
  parseStructPattern();
  adt::Result<std::shared_ptr<ast::patterns::PatternNoTopAlt>, std::string>
  parseTupleStructPattern();
  adt::Result<std::shared_ptr<ast::patterns::PatternNoTopAlt>, std::string>
  parseIdentifierPattern();
  adt::Result<std::shared_ptr<ast::patterns::PatternNoTopAlt>, std::string>
  parsePathOrStructOrTupleStructPattern();
  adt::Result<std::shared_ptr<ast::patterns::PatternNoTopAlt>, std::string>
  parseMacroInvocationPattern();
  adt::Result<std::shared_ptr<ast::patterns::PatternNoTopAlt>, std::string>
  parsePathPattern();

  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parsePathInExpressionOrStructExprStructOrStructExprTupleOrStructExprUnitOrMacroInvocationOrExpressionWithPostfix();
  adt::Result<std::shared_ptr<ast::types::TypeExpression>, std::string>
  parseTupleOrParensTypeOrTypePathOrMacroInvocationOrTraitObjectType();

  // Expressions

  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parseMacroInvocationExpression();
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parseStructExpression();

  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parsePathInExpressionOrStructExprStructOrStructTupleUnitOrMacroInvocationExpression();

  adt::Result<ast::TupleElements, std::string> parseTupleElements(Restrictions);

  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parsePathInExpressionOrStructOrExpressionWithPostfix();

  //  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  //  parseExpressionWithPostfix(std::shared_ptr<ast::Expression> left,
  //                             rust_compiler::parser::Restrictions =
  //                                 rust_compiler::parser::Restrictions());

  adt::StringResult<std::shared_ptr<ast::Expression>>
  parseExpression(std::span<ast::OuterAttribute> outer,
                  rust_compiler::parser::Restrictions restrictions);
  adt::StringResult<std::shared_ptr<ast::Expression>>
  parseExpression(Precedence rightBindingPower,
                  std::span<ast::OuterAttribute> outer,
                  rust_compiler::parser::Restrictions restrictions);

  //  llvm::Expected<std::shared_ptr<ast::Expression>>
  //  parseExpressionExceptStruct();
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseTupleExpression(Restrictions);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseGroupedExpression(Restrictions);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parseBlockExpression(std::span<ast::OuterAttribute>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseExpressionWithoutBlock(std::span<ast::OuterAttribute>, Restrictions);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parseExpressionWithoutBlockExceptStruct();
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseExpressionWithBlock(std::span<ast::OuterAttribute>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseClosureExpression(std::span<ast::OuterAttribute>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseReturnExpression(std::span<ast::OuterAttribute>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseBreakExpression(std::span<ast::OuterAttribute>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseContinueExpression(std::span<ast::OuterAttribute>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseNegationExpression(std::span<ast::OuterAttribute>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseDereferenceExpression(std::span<ast::OuterAttribute>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseBorrowExpression(std::span<ast::OuterAttribute>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseAsyncBlockExpression(std::span<ast::OuterAttribute>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseMatchExpression(std::span<ast::OuterAttribute>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseIfExpression(std::span<ast::OuterAttribute>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseIfLetExpression(std::span<ast::OuterAttribute>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseUnsafeBlockExpression(std::span<ast::OuterAttribute>);
  adt::Result<ast::Scrutinee, std::string> parseScrutinee();
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseAwaitExpression(std::shared_ptr<ast::Expression>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseIndexExpression(std::shared_ptr<ast::Expression>,
                           std::span<ast::OuterAttribute>, Restrictions);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseFieldExpression(std::shared_ptr<ast::Expression>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseRangeExpression(std::shared_ptr<ast::Expression>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parseRangeExpression();
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseUnderScoreExpression(std::span<ast::OuterAttribute>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseGroupedOrTupleExpression(Restrictions);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseCallExpression(std::shared_ptr<ast::Expression>,
                          std::span<ast::OuterAttribute>, Restrictions);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseErrorPropagationExpression(std::shared_ptr<ast::Expression>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parseAssignmentExpression(std::shared_ptr<ast::Expression>,
                            std::span<ast::OuterAttribute>,
                            Restrictions restrictions);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseTypeCastExpression(std::shared_ptr<ast::Expression>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parseMethodCallExpression(std::shared_ptr<ast::Expression>,
                            std::span<ast::OuterAttribute>,
                            Restrictions restrictions);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseTupleIndexingExpression(std::shared_ptr<ast::Expression>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseArrayExpression(std::span<ast::OuterAttribute>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parseArithmeticOrLogicalExpression(
      std::shared_ptr<ast::Expression>,
      rust_compiler::parser::Restrictions restrictions);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parseComparisonExpression(std::shared_ptr<ast::Expression>,
                            rust_compiler::parser::Restrictions restrictions =
                                rust_compiler::parser::Restrictions());
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parseLazyBooleanExpression(std::shared_ptr<ast::Expression>,
                             rust_compiler::parser::Restrictions restrictions =
                                 rust_compiler::parser::Restrictions());
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parseCompoundAssignmentExpression(
      std::shared_ptr<ast::Expression>,
      rust_compiler::parser::Restrictions restrictions);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parseQualifiedPathInExpression();
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseLiteralExpression(std::span<ast::OuterAttribute>);

  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parsePathExpression();

  adt::Result<ast::CallParams, std::string> parseCallParams(Restrictions);

  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseLoopExpression(std::span<ast::OuterAttribute>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseIteratorLoopExpression(std::span<ast::OuterAttribute>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parsePredicatePatternLoopExpression(std::span<ast::OuterAttribute>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parsePredicateLoopExpression(std::span<ast::OuterAttribute>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseInfiniteLoopExpression(std::span<ast::OuterAttribute>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseLabelBlockExpression(std::span<ast::OuterAttribute>);
  // llvm::Expected<std::shared_ptr<ast::Expression>>
  // parsePatternLoopExpression();

  adt::Result<ast::MatchArms, std::string> parseMatchArms();
  adt::Result<ast::MatchArm, std::string> parseMatchArm();
  adt::Result<ast::MatchArmGuard, std::string> parseMatchArmGuard();
  adt::Result<ast::GenericParam, std::string> parseGenericParam();

  adt::Result<ast::GenericArgs, std::string> parseGenericArgs();
  adt::Result<ast::GenericArg, std::string>
  parseGenericArg(std::optional<ast::GenericArgKind> last);
  adt::Result<ast::GenericParams, std::string> parseGenericParams();
  adt::Result<ast::GenericArgsConst, std::string> parseGenericArgsConst();
  adt::Result<ast::GenericArgsBinding, std::string> parseGenericArgsBinding();
  adt::Result<ast::WhereClause, std::string> parseWhereClause();
  adt::Result<std::shared_ptr<ast::WhereClauseItem>, std::string>
  parseWhereClauseItem();
  adt::Result<ast::types::ForLifetimes, std::string> parseForLifetimes();

  adt::Result<std::shared_ptr<ast::types::TypeParamBound>, std::string>
  parseLifetimeAsTypeParamBound();
  adt::Result<ast::Lifetime, std::string> parseLifetimeAsLifetime();
  adt::Result<ast::LifetimeBounds, std::string> parseLifetimeBounds();
  adt::Result<std::shared_ptr<ast::WhereClauseItem>, std::string>
  parseLifetimeWhereClauseItem();
  adt::Result<std::shared_ptr<ast::WhereClauseItem>, std::string>
  parseTypeBoundWhereClauseItem();

  adt::Result<ast::EnumItems, std::string> parseEnumItems();
  adt::Result<ast::EnumItem, std::string> parseEnumItem();
  adt::Result<ast::EnumItemTuple, std::string> parseEnumItemTuple();
  adt::Result<ast::EnumItemStruct, std::string> parseEnumItemStruct();
  adt::Result<ast::EnumItemDiscriminant, std::string>
  parseEnumItemDiscriminant();

  // statements
  adt::Result<std::shared_ptr<ast::Statement>, std::string>
      parseLetStatement(std::span<ast::OuterAttribute>, Restrictions);
  adt::Result<std::shared_ptr<ast::Statement>, std::string>
  parseItemDeclaration();
  adt::Result<std::shared_ptr<ast::Statement>, std::string>
      parseExpressionStatement(std::span<ast::OuterAttribute>, Restrictions);
  adt::Result<std::shared_ptr<ast::Statement>, std::string>
  parseMacroInvocationSemiStatement();

  adt::Result<std::shared_ptr<ast::Item>, std::string>
  parseMacroInvocationSemiItem();

  adt::Result<std::shared_ptr<ast::Item>, std::string>
  parseMacroRulesDefinition();
  adt::Result<std::shared_ptr<ast::SelfParam>, std::string>
  parseShorthandSelf();
  adt::Result<std::shared_ptr<ast::SelfParam>, std::string> parseTypedSelf();

  adt::Result<ast::types::TypeParamBounds, std::string> parseTypeParamBounds();
  adt::Result<std::shared_ptr<ast::types::TypeParamBound>, std::string>
  parseTypeParamBound();

  adt::Result<ast::SimplePath, std::string> parseSimplePath();
  adt::Result<ast::SimplePathSegment, std::string> parseSimplePathSegment();

  adt::Result<ast::AttrInput, std::string> parseAttrInput();

  adt::Result<std::shared_ptr<ast::DelimTokenTree>, std::string>
  parseDelimTokenTree();
  adt::Result<ast::TokenTree, std::string> parseTokenTree();

  adt::Result<ast::AssociatedItem, std::string> parseAssociatedItem();

  adt::Result<ast::Abi, std::string> parseAbi();
  adt::Result<ast::Attr, std::string> parseAttr();

  adt::Result<ast::TupleFields, std::string> parseTupleFields();
  adt::Result<ast::TupleField, std::string> parseTupleField();

  adt::Result<ast::ClosureParameters, std::string> parseClosureParameters();
  adt::Result<ast::ClosureParam, std::string> parseClosureParam();

  adt::Result<ast::types::FunctionParametersMaybeNamedVariadic, std::string>
  parseFunctionParametersMaybeNamedVariadic();

  adt::Result<ast::types::MaybeNamedParam, std::string> parseMaybeNamedParam();
  adt::Result<ast::StructExprFields, std::string> parseStructExprFields();

  adt::Result<ast::StructExprField, std::string> parseStructExprField();
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parseStructExprStruct();
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parseStructExprTuple();
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parseStructExprUnit();
  adt::Result<ast::StructBase, std::string> parseStructBase();

  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parsePathInExpression();

  adt::Result<ast::patterns::StructPatternElements, std::string>
  parseStructPatternElements();
  adt::Result<ast::patterns::StructPatternFields, std::string>
  parseStructPatternFields();
  adt::Result<ast::patterns::StructPatternField, std::string>
  parseStructPatternField();

  adt::Result<ast::patterns::TupleStructItems, std::string>
  parseTupleStructItems();

  adt::Result<ast::patterns::StructPatternEtCetera, std::string>
  parseStructPatternEtCetera();

  adt::Result<ast::ArrayElements, std::string>
  parseArrayElements(std::span<ast::OuterAttribute>, Restrictions restrictions);

  adt::Result<ast::types::FunctionParametersMaybeNamedVariadic, std::string>
  parseMaybeNamedFunctionParameters();
  adt::Result<ast::types::FunctionParametersMaybeNamedVariadic, std::string>
  parseMaybeNamedFunctionParametersVariadic();

  adt::Result<std::shared_ptr<ast::types::TypeExpression>, std::string>
  parseMacroInvocationType();

  adt::Result<std::shared_ptr<ast::patterns::PatternNoTopAlt>, std::string>
  parseGroupedOrTuplePattern();

  adt::Result<std::shared_ptr<ast::patterns::PatternNoTopAlt>, std::string>
  parseMacroInvocationOrPathOrStructOrTupleStructPattern();

  adt::Result<std::shared_ptr<ast::patterns::PatternNoTopAlt>, std::string>
  parseTuplePattern();
  adt::Result<std::shared_ptr<ast::patterns::PatternNoTopAlt>, std::string>
  parseGroupedPattern();

  adt::Result<ast::patterns::TuplePatternItems, std::string>
  parseTuplePatternItems();

  adt::Result<ast::MacroRulesDef, std::string> parseMacroRulesDef();
  adt::Result<ast::MacroRules, std::string> parseMacroRules();
  adt::Result<ast::MacroRule, std::string> parseMacroRule();
  adt::Result<ast::MacroMatcher, std::string> parseMacroMatcher();
  adt::Result<ast::MacroMatch, std::string> parseMacroMatch();
  adt::Result<ast::MacroTranscriber, std::string> parseMacroTranscriber();
  adt::Result<ast::MacroFragSpec, std::string> parseMacroFragSpec();
  adt::Result<ast::MacroRepSep, std::string> parseMacroRepSep();
  adt::Result<ast::MacroRepOp, std::string> parseMacroRepOp();

  adt::Result<std::shared_ptr<ast::types::TypeExpression>, std::string>
  parseTraitObjectTypeOrTypePathOrMacroInvocation();

  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parsePathInExpressionOrMacroInvocationExpression();

  adt::Result<ast::LoopLabel, std::string> parseLoopLabel();

  void pushFunction(std::string_view);
  void popFunction(std::string_view);

private:
  bool check(lexer::TokenKind token);
  bool check(lexer::TokenKind token, size_t off);
  bool checkKeyWord(lexer::KeyWordKind keyword);
  bool checkKeyWord(lexer::KeyWordKind keyword, size_t off);
  bool checkInKeyWords(std::span<lexer::KeyWordKind> keywords);

  bool checkOuterAttribute(uint8_t off = 0);
  bool checkInnerAttribute();

  bool checkLoopLabel();

  bool checkLiteral(uint8_t off = 0);
  bool checkLifetime(uint8_t off = 0);

  /// super | self | Self | crate | $crate
  bool checkSuperSelf();

  /// IDENTIFIER | super | self | Self | crate | $crate
  bool checkPathIdentSegment(uint8_t off = 0);
  bool eatPathIdentSegment();

  /// IDENTIFIER | super | self | crate | $crate
  bool checkSimplePathSegment();
  bool eatSimplePathSegment();

  bool checkSelfParam();

  bool eat(lexer::TokenKind token);
  bool eatKeyWord(lexer::KeyWordKind keyword);

  lexer::Token getToken(uint8_t off = 0);

  CheckPoint getCheckPoint();
  void recover(const CheckPoint &cp);

  // bool checkWhereClauseItem();
  // bool checkLifetime();

  bool checkExpressionWithBlock();
  bool checkExpressionWithoutBlock();
  bool checkExpressionWithoutBlock(std::shared_ptr<ast::Expression>);
  bool checkStatement();
  bool checkStaticOrUnderscore();
  bool checkVisItem();

  // precision could be improved
  bool checkMacroItem();
  // bool checkTypeParamBound();
  bool checkTraitBound();
  bool checkIdentifier();
  bool checkIntegerLiteral();
  bool checkMaybeNamedParamLeadingComma();

  bool checkRangeTerminator();

  bool checkDelimiters();

  bool checkPathOrStructOrMacro();
  bool checkPathExprSegment(uint8_t off = 0);
  bool checkPostFix();

  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parseBinaryExpression(bool allowBlocks);

  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parseUnaryExpression(std::span<ast::OuterAttribute> outer,
                       Restrictions restrictions);

  void printFunctionStack();
  std::stack<std::string> functionStack;

  ConstTokenPointer getTokenPtr();

  adt::StringResult<std::shared_ptr<ast::Expression>>
  parseInfixExpression(std::shared_ptr<ast::Expression> left,
                       std::span<ast::OuterAttribute>, Restrictions);
};

} // namespace rust_compiler::parser

/*
  Lifetime :
      LIFETIME_OR_LABEL
   | 'static
   | '_
 */
