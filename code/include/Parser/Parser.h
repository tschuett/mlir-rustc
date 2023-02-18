#pragma once

#include "AST/Abi.h"
#include "AST/AssociatedItem.h"
#include "AST/ConstParam.h"
#include "AST/Crate.h"
#include "AST/EnumItem.h"
#include "AST/EnumItems.h"
#include "AST/ExternalItem.h"
#include "AST/Function.h"
#include "AST/FunctionParam.h"
#include "AST/FunctionParamPattern.h"
#include "AST/FunctionParameters.h"
#include "AST/GenericParam.h"
#include "AST/LifetimeParam.h"
#include "AST/MatchArm.h"
#include "AST/MatchArmGuard.h"
#include "AST/MatchArms.h"
#include "AST/OuterAttribute.h"
#include "AST/Patterns/Pattern.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Patterns/SlicePatternItems.h"
#include "AST/Scrutinee.h"
#include "AST/SelfParam.h"
#include "AST/Statements.h"
#include "AST/TypeAlias.h"
#include "AST/TypeParam.h"
#include "AST/Types/QualifiedPathType.h"
#include "AST/Types/TraitBound.h"
#include "AST/Types/TypeParamBounds.h"
#include "AST/Types/TypePathFnInputs.h"
#include "AST/UseTree.h"
#include "AST/VisItem.h"
#include "AST/Visiblity.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Lexer/TokenStream.h"
#include "Location.h"

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
  parseImplementation(std::optional<ast::Visibility> vis);

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

  // Types
  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
  parseTypeNoBounds();
  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> parseType();
  llvm::Expected<std::shared_ptr<ast::types::TypePath>> parseTypePath();

  llvm::Expected<ast::types::TypePathSegment> parseTypePathSegment();
  llvm::Expected<ast::PathIdentSegment> parsePathIdentSegment();

  llvm::Expected<ast::types::TypePathFn> parseTypePathFn();
  llvm::Expected<ast::types::TypePathFnInputs> parseTypePathFnInputs();

  llvm::Expected<ast::types::TraitBound> parseTraitBound();
  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
  parseBareFunctionType();
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
  llvm::Expected<std::vector<std::shared_ptr<ast::Item>>> parseItems();
  // only for types !!!

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
  parseQualifiedPathInType();
  llvm::Expected<ast::types::QualifiedPathType> parseQualifiedPathType();

  PathKind testTypePathOrSimplePath();

  llvm::Expected<std::shared_ptr<ast::Crate>>
  parseCrateModule(std::string_view crateName);

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
  parsePathOrStructOrTuplePattern();
  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  parseMacroInvocation();
  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  parsePathPattern();

  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  parsePathInExpressionOrStructExprStructOrStructExprUnitOrMacroInvocation();
  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
  parseTupleOrParensTypeOrTypePathOrMacroInvocationOrTraitObjectType();

  // Expressions

  llvm::Expected<std::shared_ptr<ast::Expression>> parseExpressionWithPostfix();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseExpression();
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
      parseIndexingExpression(std::shared_ptr<ast::Expression>);
  llvm::Expected<std::shared_ptr<ast::Expression>>
      parseFieldExpression(std::shared_ptr<ast::Expression>);
  llvm::Expected<std::shared_ptr<ast::Expression>> parseRangeExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseUnderScoreExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>>
  parseGroupedOrTupleExpression();
  llvm::Expected<std::shared_ptr<ast::Expression>> parseCallExpression();
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

  llvm::Expected<ast::MatchArms> parseMatchArms();
  llvm::Expected<ast::MatchArm> parseMatchArm();
  llvm::Expected<ast::MatchArmGuard> parseMatchGuard();
  llvm::Expected<ast::GenericParam> parseGenericParam();

  llvm::Expected<ast::GenericArgs> parseGenericArgs();
  llvm::Expected<ast::GenericParams> parseGenericParams();
  llvm::Expected<ast::WhereClause> parseWhereClause();
  llvm::Expected<ast::types::ForLifetimes> parseForLifetimes();

  llvm::Expected<std::shared_ptr<ast::EnumItems>> parseEnumItems();
  llvm::Expected<std::shared_ptr<ast::EnumItem>> parseEnumItem();

  // statements
  llvm::Expected<std::shared_ptr<ast::Statement>> parseLetStatement();

  llvm::Expected<std::shared_ptr<ast::SelfParam>> parseShorthandSelf();
  llvm::Expected<std::shared_ptr<ast::SelfParam>> parseTypedSelf();

  llvm::Expected<ast::types::TypeParamBounds> parseTypeParamBounds();
  llvm::Expected<ast::types::TypeParamBound> parseTypeParamBound();

  llvm::Expected<ast::SimplePath> parseSimplePath();

  llvm::Expected<std::shared_ptr<ast::AssociatedItem>> parseAssociatedItem();

  llvm::Expected<ast::Abi> parseAbi();

private:
  bool check(lexer::TokenKind token);
  bool check(lexer::TokenKind token, size_t offset);
  bool checkKeyWord(lexer::KeyWordKind keyword);
  bool checkKeyWord(lexer::KeyWordKind keyword, size_t offset);
  bool checkInKeyWords(std::span<lexer::KeyWordKind> keywords);

  bool checkOuterAttribute();
  bool checkInnerAttribute();

  bool checkLoopLabel();

  bool checkLiteral();
  bool checkLifetime(size_t offset);

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

  lexer::Token getToken();

  CheckPoint getCheckPoint();
  void recover(const CheckPoint &cp);
};

} // namespace rust_compiler::parser

/*
  Lifetime :
      LIFETIME_OR_LABEL
   | 'static
   | '_
 */
