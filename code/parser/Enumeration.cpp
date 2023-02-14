#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <cassert>

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::EnumItem>> Parser::parseEnumItem() {}

llvm::Expected<std::shared_ptr<ast::EnumItems>> Parser::parseEnumItems() {}

llvm::Expected<std::shared_ptr<ast::VisItem>>
Parser::parseEnumeration(std::optional<ast::Visibility> vis) {
  if (!checkKeyWord(lexer::KeyWordKind::KW_ENUM)) {
    // Error
  }

  assert(eatKeyWord(KeyWordKind::KW_ENUM));

  if (!check(TokenKind::Identifier)) {
    // error
  }

  std::string identifier = getToken().getIdentifier();
  assert(eat(TokenKind::Identifier));

  if (check(TokenKind::Lt)) {
    // GenericParams
    llvm::Expected<std::shared_ptr<ast::GenericParams>> genericParams =
        parseGenericParams();
  }

  if (checkKeyWord(KeyWordKind::KW_WHERE)) {
    // where clause
    llvm::Expected<std::shared_ptr<ast::WhereClause>> whereClasue =
        parseWhereClause();
  }

  if (!check(TokenKind::BraceOpen)) {
    // error
  }

  assert(eat(TokenKind::BraceOpen));

  parseEnumItems();

  if (!check(TokenKind::BraceClose)) {
    // error
  }

  assert(eat(TokenKind::BraceClose));
}

} // namespace rust_compiler::parser
