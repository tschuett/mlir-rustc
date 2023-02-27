#pragma once

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
#include "AST/GenericParam.h"
#include "AST/Lifetime.h"
#include "AST/LifetimeBounds.h"
#include "AST/LifetimeParam.h"
#include "AST/LifetimeWhereClauseItem.h"
#include "AST/LoopExpression.h"
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

#include <_types/_uint8_t.h>
#include <llvm/Support/Error.h>
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

  llvm::Expected<ast::Visibility> parseVisibility();
  llvm::Expected<ast::use_tree::UseTree> parseUseTree();

  llvm::Expected<std::shared_ptr<ast::Item>> parseItem();
  llvm::Expected<std::shared_ptr<ast::MacroItem>> parseMacroItem();

  llvm::Expected<std::shared_ptr<ast::VisItem>> parseVisItem();

  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  parseRangeOrIdentifierOrStructOrTupleStructOrMacroInvocationPattern();

  /// VisItems
  llvm::Expected<std::shared_ptr<ast::VisItem>>
  parseMod(std::optional<ast::Visibility> vis);
  llvm::Expected<std::shared_ptr<ast::VisItem>>
  parseUseDeclaration(std::optional<ast::Visibility> vis);
  llvm::Expected<std::shared_ptr<ast::VisItem>>
  parseTypeAlias(std::optional<ast::Visibility> vis);
  llvm::Expected<std::shared_ptr<ast::VisItem>>
  parseEnumeration(std::optional<ast::Visibility> vis);
  llvm::Expected<std::shared_ptr<ast::VisItem>>
  parseUnion(std::optional<ast::Visibility> vis);
  llvm::Expected<std::shared_ptr<ast::VisItem>>
  parseStaticItem(std::optional<ast::Visibility> vis);
  llvm::Expected<std::shared_ptr<ast::VisItem>>
  parseTrait(std::optional<ast::Visibility> vis);
  llvm::Expected<std::shared_ptr<ast::VisItem>>
  parseFunction(std::optional<ast::Visibility> vis);
  llvm::Expected<std::shared_ptr<ast::VisItem>>
  parseConstantItem(std::optional<ast::Visibility> vis);
  llvm::Expected<std::shared_ptr<ast::VisItem>>
  parseExternBlock(std::optional<ast::Visibility> vis);

  llvm::Expected<ast::ExternalItem> parseExternalItem();

  llvm::Expected<std::shared_ptr<ast::VisItem>>
  parseStruct(std::optional<ast::Visibility> vis);
  llvm::Expected<ast::StructFields> parseStructFields();
  llvm::Expected<ast::StructField> parseStructField();
  llvm::Expected<std::shared_ptr<ast::VisItem>>
  parseStructStruct(std::optional<ast::Visibility> vis);
  llvm::Expected<std::shared_ptr<ast::VisItem>>
  parseTupleStruct(std::optional<ast::Visibility> vis);

  llvm::Expected<std::shared_ptr<ast::VisItem>>
  parseImplementation(std::optional<ast::Visibility> vis);
  llvm::Expected<std::shared_ptr<ast::VisItem>>
  parseInherentImpl(std::optional<ast::Visibility> vis);
  llvm::Expected<std::shared_ptr<ast::VisItem>>
  parseTraitImpl(std::optional<ast::Visibility> vis);

  llvm::Expected<std::vector<ast::OuterAttribute>> parseOuterAttributes();
  llvm::Expected<std::vector<ast::InnerAttribute>> parseInnerAttributes();

  llvm::Expected<ast::OuterAttribute> parseOuterAttribute();
  llvm::Expected<ast::InnerAttribute> parseInnerAttribute();

  llvm::Expected<ast::ConstParam> parseConstParam();
  llvm::Expected<ast::LifetimeParam> parseLifetimeParam();
  llvm::Expected<ast::TypeParam> parseTypeParam();

  // Function
  llvm::Expected<ast::FunctionQualifiers> parseFunctionQualifiers();
  // llvm::Expected<ast::FunctionSignature> parseFunctionsignature();
  llvm::Expected<ast::FunctionParam> parseFunctionParam();
  llvm::Expected<ast::FunctionParameters> parseFunctionParameters();
  llvm::Expected<ast::FunctionParamPattern> parseFunctionParamPattern();
  llvm::Expected<ast::SelfParam> parseSelfParam();

  llvm::Expected<ast::Statements> parseStatements();
  llvm::Expected<std::shared_ptr<ast::Statement>> parseStatement();

  // Types
  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
  parseTypeNoBounds();
  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> parseType();
  llvm::Expected<std::shared_ptr<ast::types::TypePath>> parseTypePath();

  llvm::Expected<ast::types::TypePathSegment> parseTypePathSegment();
  llvm::Expected<ast::PathIdentSegment> parsePathIdentSegment();

  llvm::Expected<ast::PathExprSegment> parsePathExprSegment();

  llvm::Expected<ast::types::TypePathFn> parseTypePathFn();
  llvm::Expected<ast::types::TypePathFnInputs> parseTypePathFnInputs();

  llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>> parseTraitBound();
  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
  parseBareFunctionType();
  llvm::Expected<ast::types::FunctionTypeQualifiers>
  parseFunctionTypeQualifiers();
  llvm::Expected<ast::types::BareFunctionReturnType>
  parseBareFunctionReturnType();

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
  parseTraitObjectTypeOneBound();
  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
  parseImplTraitTypeOneBound();

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
  parseImplTraitType();

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
  parseRawPointerType();
  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> parseNeverType();
  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
  parseReferenceType();
  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
  parseInferredType();
  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
  parseTraitObjectType();
  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> parseImplType();
  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
  parseArrayOrSliceType();
  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
  parseTupleOrParensType();
  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> parseTupleType();
  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
  parseTupleOrParensTypeOrTypePathOrMacroInvocationOrTraitObjectTypeOrBareFunctionType();
  // only for types !!!

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
  parseQualifiedPathInType();
  llvm::Expected<ast::types::QualifiedPathType> parseQualifiedPathType();

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
  parseParenthesizedType();

  PathKind testTypePathOrSimplePath();

  llvm::Expected<std::shared_ptr<ast::Crate>>
  parseCrateModule(std::string_view crateName, basic::CrateNum crateNum);

  // Patterns
  llvm::Expected<std::shared_ptr<ast::patterns::Pattern>> parsePattern();
  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  parsePatternNoTopAlt();
  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  parseReferencePattern();
  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  parseRestPattern();
  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  parseWildCardPattern();
  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  parseTupleOrGroupedPattern();
  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  parseSlicePattern();
  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  parseLiteralPattern();
  llvm::Expected<ast::patterns::SlicePatternItems> parseSlicePatternItems();
  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  parsePatternWithoutRange();

  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  parseRangePattern();
  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  parseStructPattern();
  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  parseTupleStructPattern();
  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  parseIdentifierPattern();
  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  parsePathOrStructOrTupleStructPattern();
  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  parseMacroInvocationPattern();
  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  parsePathPattern();

  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  parsePathInExpressionOrStructExprStructOrStructExprUnitOrMacroInvocation();
  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
  parseTupleOrParensTypeOrTypePathOrMacroInvocationOrTraitObjectType();

  // Expressions

  llvm::Expected<std::shared_ptr<ast::Expression>>
  parseMacroInvocationExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseStructExpression();

  llvm::Expected<std::shared_ptr<ast::Expression>>
  parsePathInExpressionOrStructExprStructOrStructTupleUnitOrMacroInvocationExpression();

  llvm::Expected<ast::TupleElements> parseTupleElements();

  llvm::Expected<std::shared_ptr<ast::Expression>> parseExpressionWithPostfix();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseTupleExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseGroupedExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseBlockExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>>
  parseExpressionWithoutBlock();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseExpressionWithBlock();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseClosureExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseReturnExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseBreakExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseContinueExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseNegationExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseDereferenceExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseBorrowExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseAsyncBlockExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseMatchExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseIfExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseIfLetExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseUnsafeBlockExpression();
  llvm::Expected<ast::Scrutinee> parseScrutinee();
  llvm::Expected<std::shared_ptr<ast::Expression>>
      parseAwaitExpression(std::shared_ptr<ast::Expression>);
  llvm::Expected<std::shared_ptr<ast::Expression>>
      parseIndexExpression(std::shared_ptr<ast::Expression>);
  llvm::Expected<std::shared_ptr<ast::Expression>>
      parseFieldExpression(std::shared_ptr<ast::Expression>);
  llvm::Expected<std::shared_ptr<ast::Expression>>
      parseRangeExpression(std::shared_ptr<ast::Expression>);
  llvm::Expected<std::shared_ptr<ast::Expression>> parseRangeExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseUnderScoreExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>>
  parseGroupedOrTupleExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>>
      parseCallExpression(std::shared_ptr<ast::Expression>);
  llvm::Expected<std::shared_ptr<ast::Expression>>
      parseErrorPropagationExpression(std::shared_ptr<ast::Expression>);
  llvm::Expected<std::shared_ptr<ast::Expression>>
      parseAssignmentExpression(std::shared_ptr<ast::Expression>);
  llvm::Expected<std::shared_ptr<ast::Expression>>
      parseTypeCastExpression(std::shared_ptr<ast::Expression>);
  llvm::Expected<std::shared_ptr<ast::Expression>>
      parseMethodCallExpression(std::shared_ptr<ast::Expression>);
  llvm::Expected<std::shared_ptr<ast::Expression>>
      parseTupleIndexingExpression(std::shared_ptr<ast::Expression>);
  llvm::Expected<std::shared_ptr<ast::Expression>> parseArrayExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>>
      parseArithmeticOrLogicalExpression(std::shared_ptr<ast::Expression>);
  llvm::Expected<std::shared_ptr<ast::Expression>>
      parseComparisonExpression(std::shared_ptr<ast::Expression>);
  llvm::Expected<std::shared_ptr<ast::Expression>>
      parseLazyBooleanExpression(std::shared_ptr<ast::Expression>);
  llvm::Expected<std::shared_ptr<ast::Expression>>
      parseCompoundAssignmentExpression(std::shared_ptr<ast::Expression>);
  llvm::Expected<std::shared_ptr<ast::Expression>>
  parseQualifiedPathInExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseLiteralExpression();

  llvm::Expected<std::shared_ptr<ast::Expression>> parsePathExpression();

  llvm::Expected<ast::CallParams> parseCallParams();

  llvm::Expected<std::shared_ptr<ast::Expression>> parseLoopExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>>
  parseIteratorLoopExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>>
  parsePredicatePatternLoopExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>>
  parsePredicateLoopExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>>
  parseInfiniteLoopExpression();
  // llvm::Expected<std::shared_ptr<ast::Expression>>
  // parsePatternLoopExpression();

  llvm::Expected<ast::MatchArms> parseMatchArms();
  llvm::Expected<ast::MatchArm> parseMatchArm();
  llvm::Expected<ast::MatchArmGuard> parseMatchArmGuard();
  llvm::Expected<ast::GenericParam> parseGenericParam();

  llvm::Expected<ast::GenericArgs> parseGenericArgs();
  llvm::Expected<ast::GenericParams> parseGenericParams();
  llvm::Expected<ast::WhereClause> parseWhereClause();
  llvm::Expected<std::shared_ptr<ast::WhereClauseItem>> parseWhereClauseItem();
  llvm::Expected<ast::types::ForLifetimes> parseForLifetimes();

  llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>> parseLifetimeAsTypeParamBound();
  llvm::Expected<ast::Lifetime> parseLifetimeAsLifetime();
  llvm::Expected<ast::LifetimeBounds> parseLifetimeBounds();
  llvm::Expected<std::shared_ptr<ast::WhereClauseItem>>
  parseLifetimeWhereClauseItem();
  llvm::Expected<std::shared_ptr<ast::WhereClauseItem>>
  parseTypeBoundWhereClauseItem();

  llvm::Expected<ast::EnumItems> parseEnumItems();
  llvm::Expected<ast::EnumItem> parseEnumItem();
  llvm::Expected<ast::EnumItemTuple> parseEnumItemTuple();
  llvm::Expected<ast::EnumItemStruct> parseEnumItemStruct();
  llvm::Expected<ast::EnumItemDiscriminant> parseEnumItemDiscriminant();

  // statements
  llvm::Expected<std::shared_ptr<ast::Statement>> parseLetStatement();
  llvm::Expected<std::shared_ptr<ast::Statement>> parseItemDeclaration();
  llvm::Expected<std::shared_ptr<ast::Statement>> parseExpressionStatement();
  llvm::Expected<std::shared_ptr<ast::Statement>>
  parseMacroInvocationSemiStatement();

  llvm::Expected<std::shared_ptr<ast::MacroItem>>
  parseMacroInvocationSemiItem();

  llvm::Expected<std::shared_ptr<ast::MacroItem>> parseMacroRulesDefinition();
  llvm::Expected<std::shared_ptr<ast::SelfParam>> parseShorthandSelf();
  llvm::Expected<std::shared_ptr<ast::SelfParam>> parseTypedSelf();

  llvm::Expected<ast::types::TypeParamBounds> parseTypeParamBounds();
  llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>>
  parseTypeParamBound();

  llvm::Expected<ast::SimplePath> parseSimplePath();

  llvm::Expected<ast::AttrInput> parseAttrInput();

  llvm::Expected<std::shared_ptr<ast::DelimTokenTree>> parseDelimTokenTree();
  llvm::Expected<ast::TokenTree> parseTokenTree();

  llvm::Expected<ast::AssociatedItem> parseAssociatedItem();

  llvm::Expected<ast::Abi> parseAbi();
  llvm::Expected<ast::Attr> parseAttr();

  llvm::Expected<ast::TupleFields> parseTupleFields();
  llvm::Expected<ast::TupleField> parseTupleField();

  llvm::Expected<ast::ClosureParameters> parseClosureParameters();
  llvm::Expected<ast::ClosureParam> parseClosureParam();

  llvm::Expected<ast::types::FunctionParametersMaybeNamedVariadic>
  parseFunctionParametersMaybeNamedVariadic();

  llvm::Expected<ast::types::MaybeNamedParam> parseMaybeNamedParam();
  llvm::Expected<ast::StructExprFields> parseStructExprFields();

  llvm::Expected<ast::StructExprField> parseStructExprField();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseStructExprStruct();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseStructExprTuple();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseStructExprUnit();
  llvm::Expected<ast::StructBase> parseStructBase();

  llvm::Expected<std::shared_ptr<ast::PathExpression>> parsePathInExpression();

  llvm::Expected<ast::patterns::StructPatternElements>
  parseStructPatternElements();
  llvm::Expected<ast::patterns::StructPatternFields> parseStructPatternFields();
  llvm::Expected<ast::patterns::StructPatternField> parseStructPatternField();

  llvm::Expected<ast::patterns::TupleStructItems> parseTupleStructItems();

  llvm::Expected<ast::patterns::StructPatternEtCetera>
  parseStructPatternEtCetera();

  llvm::Expected<ast::ArrayElements> parseArrayElements();

  llvm::Expected<ast::types::FunctionParametersMaybeNamedVariadic>
  parseMaybeNamedFunctionParameters();
  llvm::Expected<ast::types::FunctionParametersMaybeNamedVariadic>
  parseMaybeNamedFunctionParametersVariadic();

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
  parseMacroInvocationType();

  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  parseGroupedOrTuplePattern();

  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  parseMacroInvocationOrPathOrStructOrTupleStructPattern();

  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  parseTuplePattern();
  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  parseGroupedPattern();

  llvm::Expected<ast::patterns::TuplePatternItems> parseTuplePatternItems();

  llvm::Expected<ast::MacroRulesDef> parseMacroRulesDef();
  llvm::Expected<ast::MacroRules> parseMacroRules();
  llvm::Expected<ast::MacroRule> parseMacroRule();
  llvm::Expected<ast::MacroMatcher> parseMacroMatcher();
  llvm::Expected<ast::MacroMatch> parseMacroMatch();
  llvm::Expected<ast::MacroTranscriber> parseMacroTranscriber();
  llvm::Expected<ast::MacroFragSpec> parseMacroFragSpec();
  llvm::Expected<ast::MacroRepSep> parseMacroRepSep();
  llvm::Expected<ast::MacroRepOp> parseMacroRepOp();

private:
  bool check(lexer::TokenKind token);
  bool check(lexer::TokenKind token, size_t offset);
  bool checkKeyWord(lexer::KeyWordKind keyword);
  bool checkKeyWord(lexer::KeyWordKind keyword, size_t offset);
  bool checkInKeyWords(std::span<lexer::KeyWordKind> keywords);

  bool checkOuterAttribute(uint8_t offset = 0);
  bool checkInnerAttribute();

  bool checkLoopLabel();

  bool checkLiteral();
  bool checkLifetime(uint8_t offset = 0);

  /// super | self | Self | crate | $crate
  bool checkSuperSelf();

  /// IDENTIFIER | super | self | Self | crate | $crate
  bool checkPathIdentSegment();
  bool eatPathIdentSegment();

  /// IDENTIFIER | super | self | crate | $crate
  bool checkSimplePathSegment();
  bool eatSimplePathSegment();

  bool checkSelfParam();

  bool eat(lexer::TokenKind token);
  bool eatKeyWord(lexer::KeyWordKind keyword);

  lexer::Token getToken(uint8_t = 0);

  CheckPoint getCheckPoint();
  void recover(const CheckPoint &cp);

  bool checkWhereClauseItem();
  //bool checkLifetime();

  bool checkExpressionWithBlock();
  bool checkExpressionWithoutBlock();
  bool checkStatement();
  bool checkStaticOrUnderscore();
  bool checkVisItem();
  bool checkMacroItem();
  //bool checkTypeParamBound();
  bool checkTraitBound();
  bool checkIdentifier();
  bool checkIntegerLiteral();
  bool checkMaybeNamedParamLeadingComma();

  bool checkRangeTerminator();

  bool checkDelimiters();
};

} // namespace rust_compiler::parser

/*
  Lifetime :
      LIFETIME_OR_LABEL
   | 'static
   | '_
 */
