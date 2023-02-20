#include "AST/Enumeration.h"

#include "AST/EnumItemDiscriminant.h"
#include "AST/EnumItemStruct.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <cassert>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace llvm;

namespace rust_compiler::parser {

llvm::Expected<ast::EnumItem> Parser::parseEnumItem() {
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

  if (check(TokenKind::BraceOpen)) {
    // Struct
    llvm::Expected<ast::EnumItemStruct> struc = parseEnumItemStruct();
    if (auto e = struc.takeError()) {
      llvm::errs() << "failed to parse enum item struct in enum item: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    item.setEnumItemStruct(*struc);
  } else if (check(TokenKind::Eq)) {
    // Dis
    llvm::Expected<ast::EnumItemDiscriminant> dis = parseEnumItemDiscriminant();
    if (auto e = dis.takeError()) {
      llvm::errs() << "failed to parse enum item discriminant in enum item: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    item.setEnumItemDiscriminant(*dis);
  } else if (check(TokenKind::ParenOpen)) {
    // Tupl
    llvm::Expected<ast::EnumItemTuple> tupl = parseEnumItemTuple();
    if (auto e = tupl.takeError()) {
      llvm::errs() << "failed to parse enum item tuple in enum item: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    item.setEnumItemTuple(*tupl);
  } else if (check(TokenKind::Comma)) {
    // done?
  } else {
    // done ?
  }

  return item;
}

llvm::Expected<std::shared_ptr<ast::VisItem>>
Parser::parseEnumeration(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();

  Enumeration enu = {loc, vis};

  if (!checkKeyWord(lexer::KeyWordKind::KW_ENUM)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse enum keyword in enum ");
  }
  assert(eatKeyWord(KeyWordKind::KW_ENUM));

  if (!check(TokenKind::Identifier)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse identifier token in enum ");
  }

  enu.setIdentifier(getToken().getIdentifier());
  assert(eat(TokenKind::Identifier));

  if (check(TokenKind::Lt)) {
    // GenericParams
    llvm::Expected<ast::GenericParams> genericParams = parseGenericParams();
    if (auto e = genericParams.takeError()) {
      llvm::errs() << "failed to parse generic params in enum: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    enu.setGenericParams(*genericParams);
  }

  if (checkKeyWord(KeyWordKind::KW_WHERE)) {
    llvm::Expected<ast::WhereClause> whereClause = parseWhereClause();
    if (auto e = whereClause.takeError()) {
      llvm::errs() << "failed to parse where clause in enum: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    enu.setWhereClause(*whereClause);
  }

  if (!check(TokenKind::BraceOpen)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse { token in enum ");
  }
  assert(eat(TokenKind::BraceOpen));

  if (check(TokenKind::BraceClose)) {
    // done
  } else {
    llvm::Expected<ast::EnumItems> items = parseEnumItems();
    if (auto e = items.takeError()) {
      llvm::errs() << "failed to parse enum items in enum: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    enu.setItems(*items);
  }

  if (!check(TokenKind::BraceClose)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse } token in enum ");
  }
  assert(eat(TokenKind::BraceClose));

  return std::make_shared<Enumeration>(enu);
}

llvm::Expected<ast::EnumItemTuple> Parser::parseEnumItemTuple() {
  Location loc = getLocation();

  EnumItemTuple tup = {loc};

  if (!check(TokenKind::ParenOpen))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse ( token in enum item tuple");
  assert(eat(TokenKind::ParenOpen));

  llvm::Expected<ast::TupleFields> fields = parseTupleFields();
  if (auto e = fields.takeError()) {
    llvm::errs()
        << "failed to parse tuple  fields expression in enum item tuple: "
        << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  tup.setTupleFields(*fields);

  if (!check(TokenKind::ParenClose)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse ) token in enum item tuple");
  }
  assert(eat(TokenKind::ParenClose));

  return tup;
}

llvm::Expected<ast::EnumItemStruct> Parser::parseEnumItemStruct() {
  Location loc = getLocation();

  EnumItemStruct str = {loc};

  if (!check(TokenKind::BraceOpen))
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse { token in enum item discriminant");

  assert(eat(TokenKind::BraceOpen));

  llvm::Expected<ast::StructFields> fields = parseStructFields();
  if (auto e = fields.takeError()) {
    llvm::errs()
        << "failed to parse struct fields expression in enum item struct: "
        << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  str.setStructFields(*fields);

  if (!check(TokenKind::BraceClose))
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse } token in enum item discriminant");

  assert(eat(TokenKind::BraceClose));

  return str;
}

llvm::Expected<ast::EnumItemDiscriminant> Parser::parseEnumItemDiscriminant() {
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

  return dis;
}

llvm::Expected<ast::EnumItems> Parser::parseEnumItems() {
  Location loc = getLocation();

  EnumItems items = {loc};

  llvm::Expected<ast::EnumItem> first = parseEnumItem();
  if (auto e = first.takeError()) {
    llvm::errs() << "failed to parse enum teim in enum items: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  items.addItem(*first);

  while(true) {
    if (check(TokenKind::Eof)) {
      // abort
    } else if (check(TokenKind::Comma)) {
    } else if (check(TokenKind::BraceOpen)) {
    } else if (check(TokenKind::ParenOpen)) {
    } else if (check(TokenKind::Eq)) {
    } else {
      // ?
    }
  }
}

} // namespace rust_compiler::parser
