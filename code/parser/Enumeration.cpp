#include "AST/EnumItemDiscriminant.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <cassert>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace llvm;

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::EnumItem>> Parser::parseEnumItem() {
  Location loc = getLocation();

  EnumItem item = {loc};

  if (checkOuterAttribute()) {
    llvm::Expected<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (auto e = outer.takeError()) {
      llvm::errs() << "failed to parse outer attributes in enum item: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    item.setOuterAttributes(*outer);
  }

  if (checkKeyWord(KeyWordKind::KW_PUB)) {
    llvm::Expected<ast::Visibility> vis = parseVisibility();
    if (auto e = vis.takeError()) {
      llvm::errs() << "failed to parse visiblity in enum item: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    item.setVisibility(*vis);
  }

  if (!check(TokenKind::Identifier)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse identifier token in enum item");
  }
  Token tok = getToken();
  item.setIdentifier(tok.getIdentifier());
  assert(eat(TokenKind::Identifier));

  if (check(TokenKind::ParenOpen)) {

  } else if (check(TokenKind::BraceOpen)) {
  } else if (check(TokenKind::Eq)) {
  } else {
  }
}

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
    llvm::Expected<ast::GenericParams> genericParams = parseGenericParams();
  }

  if (checkKeyWord(KeyWordKind::KW_WHERE)) {
    // where clause
    llvm::Expected<ast::WhereClause> whereClasue = parseWhereClause();
    // check error
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

llvm::Expected<std::shared_ptr<ast::EnumItemTuple>>
Parser::parseEnumItemTuple() {}

llvm::Expected<std::shared_ptr<ast::EnumItemStruct>>
Parser::parseEnumItemStruct() {}

llvm::Expected<std::shared_ptr<ast::EnumItemDiscriminant>>
Parser::parseEnumItemDiscriminant() {
  Location loc = getLocation();

  EnumItemDiscriminant dis = {loc};

  if (!check(TokenKind::Eq))
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse = token in enum item discriminant");

  assert(eat(TokenKind::Eq));

  llvm::Expected<std::shared_ptr<ast::Expression>> expr = parseExpression();
  if (auto e = expr.takeError()) {
    llvm::errs() << "failed to parse expression in enum item discriminant: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  dis.setExpression(*expr);

  return std::make_shared<EnumItemDiscriminant>(dis);
}

} // namespace rust_compiler::parser
