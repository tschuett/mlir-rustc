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

#include <llvm/Support/Error.h>
#include <span>
#include <stack>
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
  llvm::Expected<std::shared_ptr<ast::MacroItem>>
      parseMacroItem(std::span<ast::OuterAttribute>);

  llvm::Expected<std::shared_ptr<ast::VisItem>>
      parseVisItem(std::span<ast::OuterAttribute>);

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
  llvm::Expected<ast::FunctionReturnType> parseFunctionReturnType();
  llvm::Expected<ast::SelfParam> parseSelfParam();

  llvm::Expected<ast::Statements> parseStatements();
  llvm::Expected<std::shared_ptr<ast::Statement>> parseStatement();

  // Types
  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
  parseTypeNoBounds();
  adt::Result<std::shared_ptr<ast::types::TypeExpression>, std::string>
  parseType();
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

  llvm::Expected<std::shared_ptr<ast::Expression>>
  parsePathInExpressionOrStructExprStructOrStructExprTupleOrStructExprUnitOrMacroInvocationOrExpressionWithPostfix();
  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
  parseTupleOrParensTypeOrTypePathOrMacroInvocationOrTraitObjectType();

  // Expressions

  llvm::Expected<std::shared_ptr<ast::Expression>>
  parseMacroInvocationExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseStructExpression();

  llvm::Expected<std::shared_ptr<ast::Expression>>
  parsePathInExpressionOrStructExprStructOrStructTupleUnitOrMacroInvocationExpression();

  adt::Result<ast::TupleElements, std::string> parseTupleElements(Restrictions);

  llvm::Expected<std::shared_ptr<ast::Expression>>
  parsePathInExpressionOrStructOrExpressionWithPostfix();

  llvm::Expected<std::shared_ptr<ast::Expression>> parseExpressionWithPostfix(
      llvm::Expected<std::shared_ptr<ast::Expression>> left,
      rust_compiler::parser::Restrictions =
          rust_compiler::parser::Restrictions());

  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parseExpression(std::span<ast::OuterAttribute> outer,
                  rust_compiler::parser::Restrictions restrictions =
                      rust_compiler::parser::Restrictions());
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parseExpression(Precedence rightBindingPower,
                  std::span<ast::OuterAttribute> outer,
                  rust_compiler::parser::Restrictions restrictions =
                      rust_compiler::parser::Restrictions());

  //  llvm::Expected<std::shared_ptr<ast::Expression>>
  //  parseExpressionExceptStruct();
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseTupleExpression(Restrictions);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseGroupedExpression(Restrictions);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseBlockExpression(std::span<ast::OuterAttribute>);
  llvm::Expected<std::shared_ptr<ast::Expression>>
  parseExpressionWithoutBlock();
  llvm::Expected<std::shared_ptr<ast::Expression>>
  parseExpressionWithoutBlockExceptStruct();
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseExpressionWithBlock(std::span<ast::OuterAttribute>);
  llvm::Expected<std::shared_ptr<ast::Expression>> parseClosureExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseReturnExpression();
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseBreakExpression(std::span<ast::OuterAttribute>);
  llvm::Expected<std::shared_ptr<ast::Expression>> parseContinueExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseNegationExpression();
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseDereferenceExpression(std::span<ast::OuterAttribute>);
  llvm::Expected<std::shared_ptr<ast::Expression>> parseBorrowExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseAsyncBlockExpression();
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseMatchExpression(std::span<ast::OuterAttribute>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseIfExpression(std::span<ast::OuterAttribute>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseIfLetExpression(std::span<ast::OuterAttribute>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseUnsafeBlockExpression(std::span<ast::OuterAttribute>);
  adt::Result<ast::Scrutinee, std::string> parseScrutinee();
  llvm::Expected<std::shared_ptr<ast::Expression>>
      parseAwaitExpression(std::shared_ptr<ast::Expression>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseIndexExpression(std::shared_ptr<ast::Expression>,
                           std::span<ast::OuterAttribute>, Restrictions);
  llvm::Expected<std::shared_ptr<ast::Expression>>
      parseFieldExpression(std::shared_ptr<ast::Expression>);
  llvm::Expected<std::shared_ptr<ast::Expression>>
      parseRangeExpression(std::shared_ptr<ast::Expression>);
  llvm::Expected<std::shared_ptr<ast::Expression>> parseRangeExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseUnderScoreExpression();
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseGroupedOrTupleExpression(Restrictions);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseCallExpression(std::shared_ptr<ast::Expression>,
                          std::span<ast::OuterAttribute>, Restrictions);
  llvm::Expected<std::shared_ptr<ast::Expression>>
      parseErrorPropagationExpression(std::shared_ptr<ast::Expression>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parseAssignmentExpression(std::shared_ptr<ast::Expression>,
                            std::span<ast::OuterAttribute>,
                            Restrictions restrictions);
  llvm::Expected<std::shared_ptr<ast::Expression>>
      parseTypeCastExpression(std::shared_ptr<ast::Expression>);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parseMethodCallExpression(std::shared_ptr<ast::Expression>,
                            std::span<ast::OuterAttribute>,
                            Restrictions restrictions);
  llvm::Expected<std::shared_ptr<ast::Expression>>
      parseTupleIndexingExpression(std::shared_ptr<ast::Expression>);
  llvm::Expected<std::shared_ptr<ast::Expression>> parseArrayExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>>
  parseArithmeticOrLogicalExpression(
      std::shared_ptr<ast::Expression>,
      rust_compiler::parser::Restrictions restrictions);
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parseComparisonExpression(std::shared_ptr<ast::Expression>,
                            rust_compiler::parser::Restrictions restrictions =
                                rust_compiler::parser::Restrictions());
  llvm::Expected<std::shared_ptr<ast::Expression>>
  parseLazyBooleanExpression(std::shared_ptr<ast::Expression>,
                             rust_compiler::parser::Restrictions restrictions =
                                 rust_compiler::parser::Restrictions());
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
  parseCompoundAssignmentExpression(
      std::shared_ptr<ast::Expression>,
      rust_compiler::parser::Restrictions restrictions);
  llvm::Expected<std::shared_ptr<ast::Expression>>
  parseQualifiedPathInExpression();
  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseLiteralExpression(std::span<ast::OuterAttribute>);

  llvm::Expected<std::shared_ptr<ast::Expression>> parsePathExpression();

  adt::Result<ast::CallParams, std::string> parseCallParams(Restrictions);

  adt::Result<std::shared_ptr<ast::Expression>, std::string>
      parseLoopExpression(std::span<ast::OuterAttribute>);
  llvm::Expected<std::shared_ptr<ast::Expression>>
      parseIteratorLoopExpression(std::optional<std::string>);
  llvm::Expected<std::shared_ptr<ast::Expression>>
      parsePredicatePatternLoopExpression(std::optional<std::string>);
  llvm::Expected<std::shared_ptr<ast::Expression>>
      parsePredicateLoopExpression(std::optional<std::string>);
  llvm::Expected<std::shared_ptr<ast::Expression>>
      parseInfiniteLoopExpression(std::optional<std::string>);
  llvm::Expected<std::shared_ptr<ast::Expression>>
      parseLabelBlockExpression(std::optional<std::string>);
  // llvm::Expected<std::shared_ptr<ast::Expression>>
  // parsePatternLoopExpression();

  llvm::Expected<ast::MatchArms> parseMatchArms();
  llvm::Expected<ast::MatchArm> parseMatchArm();
  llvm::Expected<ast::MatchArmGuard> parseMatchArmGuard();
  llvm::Expected<ast::GenericParam> parseGenericParam();

  llvm::Expected<ast::GenericArgs> parseGenericArgs();
  llvm::Expected<ast::GenericArg>
  parseGenericArg(std::optional<ast::GenericArgKind> last);
  llvm::Expected<ast::GenericParams> parseGenericParams();
  llvm::Expected<ast::GenericArgsConst> parseGenericArgsConst();
  llvm::Expected<ast::GenericArgsBinding> parseGenericArgsBinding();
  llvm::Expected<ast::WhereClause> parseWhereClause();
  llvm::Expected<std::shared_ptr<ast::WhereClauseItem>> parseWhereClauseItem();
  llvm::Expected<ast::types::ForLifetimes> parseForLifetimes();

  llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>>
  parseLifetimeAsTypeParamBound();
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
  llvm::Expected<ast::SimplePathSegment> parseSimplePathSegment();

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

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
  parseTraitObjectTypeOrTypePathOrMacroInvocation();

  llvm::Expected<std::shared_ptr<ast::Expression>>
  parsePathInExpressionOrMacroInvocationExpression();

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

  llvm::Expected<std::shared_ptr<ast::Expression>>
  parseBinaryExpression(bool allowBlocks);

  llvm::Expected<std::shared_ptr<ast::Expression>>
  parseUnaryExpression(std::span<ast::OuterAttribute> outer,
                       ParseRestrictions restrictions = ParseRestrictions());

  void printFunctionStack();
  void pushFunction(std::string_view);
  void popFunction(std::string_view);
  std::stack<std::string> functionStack;
};

} // namespace rust_compiler::parser

/*
  Lifetime :
      LIFETIME_OR_LABEL
   | 'static
   | '_
 */
