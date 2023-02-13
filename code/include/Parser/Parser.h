#pragma once

#include "AST/Crate.h"
#include "AST/Function.h"
#include "AST/OuterAttribute.h"
#include "AST/TypeAlias.h"
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
  llvm::Expected<std::shared_ptr<ast::VisItem>>
  parseImplementation(std::optional<ast::Visibility> vis);

  llvm::Expected<std::vector<ast::OuterAttribute>> parseOuterAttributes();

  // Function
  llvm::Expected<ast::FunctionQualifiers> parseFunctionQualifiers();
  llvm::Expected<std::shared_ptr<ast::BlockExpression>> parseFunctionBody();
  llvm::Expected<ast::FunctionSignature> parseFunctionsignature();
  llvm::Expected<ast::FunctionParam> parseFunctionParam();

  llvm::Expected<std::vector<std::shared_ptr<ast::Item>>> parseItems();

  llvm::Expected<std::shared_ptr<ast::Crate>>
  parseCrateModule(std::string_view crateName);

private:
  bool check(lexer::TokenKind token);
  bool check(lexer::TokenKind token, size_t offset);
  bool checkKeyWord(lexer::KeyWordKind keyword);
  bool checkKeyWord(lexer::KeyWordKind keyword, size_t offset);
  bool checkInKeyWords(std::span<lexer::KeyWordKind> keywords);

  bool eat(lexer::TokenKind token);
  bool eatKeyWord(lexer::KeyWordKind keyword);

  lexer::Token getToken();
};

} // namespace rust_compiler::parser
