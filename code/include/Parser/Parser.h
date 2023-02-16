#pragma once

#include "AST/Crate.h"
#include "AST/EnumItem.h"
#include "AST/EnumItems.h"
#include "AST/Function.h"
#include "AST/FunctionParam.h"
#include "AST/FunctionParamPattern.h"
#include "AST/FunctionParameters.h"
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
#include "AST/Types/QualifiedPathType.h"
#include "AST/Types/TraitBound.h"
#include "AST/Types/TypeParamBounds.h"
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
  llvm::Expected<std::shared_ptr<ast::VisItem>>
  parseStruct(std::optional<ast::Visibility> vis);
  llvm::Expected<std::shared_ptr<ast::StructFields>> parseStructFields();
  llvm::Expected<std::shared_ptr<ast::StructField>> parseStructField();

  llvm::Expected<std::shared_ptr<ast::VisItem>>
  parseImplementation(std::optional<ast::Visibility> vis);

  llvm::Expected<std::vector<ast::OuterAttribute>> parseOuterAttributes();
  llvm::Expected<std::vector<ast::InnerAttribute>> parseInnerAttributes();

  llvm::Expected<ast::OuterAttribute> parseOuterAttribute();
  llvm::Expected<ast::InnerAttribute> parseInnerAttribute();

  // Function
  llvm::Expected<ast::FunctionQualifiers> parseFunctionQualifiers();
  // llvm::Expected<ast::FunctionSignature> parseFunctionsignature();
  llvm::Expected<ast::FunctionParam> parseFunctionParam();
  llvm::Expected<ast::FunctionParameters> parseFunctionParameters();
  llvm::Expected<ast::FunctionParamPattern> parseFunctionParamPattern();
  llvm::Expected<ast::SelfParam> parseSelfParam();

  llvm::Expected<ast::Statements> parseStatements();

  // Types
  llvm::Expected<std::shared_ptr<ast::types::TypeNoBounds>> parseTypeNoBounds();
  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> parseType();
  llvm::Expected<std::shared_ptr<ast::types::TypePath>> parseTypePath();
  llvm::Expected<ast::types::TypeParamBounds> parseTypeParamBounds();
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
  llvm::Expected<ast::patterns::Pattern> parsePattern();
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
  llvm::Expected<ast::patterns::SlicePatternItems> parseSlicePatternItems();
  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  parsePatternWithoutRange();

  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  parseRangePattern();
  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  parseIdentifierPattern();

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
  parseTupleOrParensTypeOrTypePathOrMacroInvocationOrTraitObjectType();

  // Expressions
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

  llvm::Expected<ast::MatchArms> parseMatchArms();
  llvm::Expected<ast::MatchArm> parseMatchArm();
  llvm::Expected<ast::MatchArmGuard> parseMatchGuard();

  llvm::Expected<ast::GenericParams> parseGenericParams();
  llvm::Expected<ast::WhereClause> parseWhereClause();
  llvm::Expected<ast::types::ForLifetimes> parseForLifetimes();

  llvm::Expected<std::shared_ptr<ast::EnumItems>> parseEnumItems();
  llvm::Expected<std::shared_ptr<ast::EnumItem>> parseEnumItem();

  // statements
  llvm::Expected<std::shared_ptr<ast::Statement>> parseLetStatement();

  llvm::Expected<std::shared_ptr<ast::SelfParam>> parseShorthandSelf();
  llvm::Expected<std::shared_ptr<ast::SelfParam>> parseTypedSelf();

  llvm::Expected<ast::SimplePath> parseSimplePath();

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

  bool checkSelfParam();

  bool eat();
  bool eat(lexer::TokenKind token);
  bool eatKeyWord(lexer::KeyWordKind keyword);

  lexer::Token getToken();

  
};

} // namespace rust_compiler::parser

/*
  Lifetime :
      LIFETIME_OR_LABEL
   | 'static
   | '_
 */
