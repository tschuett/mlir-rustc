#include "AST/Enumeration.h"

#include "ADT/Result.h"
#include "AST/EnumItemDiscriminant.h"
#include "AST/EnumItemStruct.h"
#include "AST/OuterAttribute.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"
#include "Parser/Restrictions.h"

#include <cassert>
#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace llvm;

namespace rust_compiler::parser {

StringResult<ast::EnumItem> Parser::parseEnumItem() {
  Location loc = getLocation();

  EnumItem item = {loc};

  if (checkOuterAttribute()) {
    StringResult<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (!outer) {
      llvm::errs() << "failed to parse outer attributes in enum item: "
                   << outer.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    std::vector<OuterAttribute> ot = outer.getValue();
    item.setOuterAttributes(ot);
  }

  if (checkKeyWord(KeyWordKind::KW_PUB)) {
    StringResult<ast::Visibility> vis = parseVisibility();
    if (!vis) {
      llvm::errs() << "failed to parse visibility in enum item: "
                   << vis.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    item.setVisibility(vis.getValue());
  }

  if (!check(TokenKind::Identifier)) {
    return StringResult<ast::EnumItem>(
                             "failed to parse identifier token in enum item");
  }
  Token tok = getToken();
  item.setIdentifier(tok.getIdentifier());
  assert(eat(TokenKind::Identifier));

  if (check(TokenKind::BraceOpen)) {
    // Struct
    StringResult<ast::EnumItemStruct> struc = parseEnumItemStruct();
    if (!struc) {
      llvm::errs() << "failed to parse enum item struct in enum item: "
                   << struc.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    item.setEnumItemStruct(struc.getValue());
  } else if (check(TokenKind::Eq)) {
    // Dis
    StringResult<ast::EnumItemDiscriminant> dis = parseEnumItemDiscriminant();
    if (!dis) {
      llvm::errs()
          << "failed to parse enum item discriminatn tuple in enum item: "
          << dis.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    item.setEnumItemDiscriminant(dis.getValue());
  } else if (check(TokenKind::ParenOpen)) {
    // Tupl
    StringResult<ast::EnumItemTuple> tupl = parseEnumItemTuple();
    if (!tupl) {
      llvm::errs() << "failed to parse enum item tuple in enum item: "
                   << tupl.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    item.setEnumItemTuple(tupl.getValue());
  } else if (check(TokenKind::Comma)) {
    // done?

  } else {
    // done ?
  }

  return StringResult<ast::EnumItem>(item);
}

StringResult<std::shared_ptr<ast::VisItem>>
Parser::parseEnumeration(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();

  Enumeration enu = {loc, vis};

  if (!checkKeyWord(lexer::KeyWordKind::KW_ENUM)) {
    return StringResult<std::shared_ptr<ast::VisItem>>(
        "failed to parse enum keyword in enum ");
  }
  assert(eatKeyWord(KeyWordKind::KW_ENUM));

  if (!check(TokenKind::Identifier)) {
    return StringResult<std::shared_ptr<ast::VisItem>>(
        "failed to parse identifier token in enum ");
  }

  enu.setIdentifier(getToken().getIdentifier());
  assert(eat(TokenKind::Identifier));

  if (check(TokenKind::Lt)) {
    // GenericParams
    StringResult<ast::GenericParams> genericParams = parseGenericParams();
    if (!genericParams) {
      llvm::errs() << "failed to parse generic params in enumeration: "
                   << genericParams.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    enu.setGenericParams(genericParams.getValue());
  }

  if (checkKeyWord(KeyWordKind::KW_WHERE)) {
    StringResult<ast::WhereClause> whereClause = parseWhereClause();
    if (!whereClause) {
      llvm::errs() << "failed to parse where clause in enumeration: "
                   << whereClause.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    enu.setWhereClause(whereClause.getValue());
  }

  if (!check(TokenKind::BraceOpen)) {
    return StringResult<std::shared_ptr<ast::VisItem>>(
        "failed to parse { token in enum ");
  }
  assert(eat(TokenKind::BraceOpen));

  if (check(TokenKind::BraceClose)) {
    // done
  } else {
    StringResult<ast::EnumItems> items = parseEnumItems();
    if (!items) {
      llvm::errs() << "failed to parse enum items in enumeration: "
                   << items.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    enu.setItems(items.getValue());
  }

  if (!check(TokenKind::BraceClose)) {
    return StringResult<std::shared_ptr<ast::VisItem>>(
        "failed to parse } token in enum ");
  }
  assert(eat(TokenKind::BraceClose));

  return StringResult<std::shared_ptr<ast::VisItem>>(
      std::make_shared<Enumeration>(enu));
}

StringResult<ast::EnumItemTuple> Parser::parseEnumItemTuple() {
  Location loc = getLocation();

  EnumItemTuple tup = {loc};

  if (!check(TokenKind::ParenOpen))
    return StringResult<ast::EnumItemTuple>(
        "failed to parse ( token in enum item tuple");
  assert(eat(TokenKind::ParenOpen));

  if (check(TokenKind::ParenClose)) {
    assert(eat(TokenKind::ParenClose));
    return StringResult<ast::EnumItemTuple>(tup);
  }

  StringResult<ast::TupleFields> fields = parseTupleFields();
  if (!fields) {
    llvm::errs() << "failed to parse tuple fields in enum item tuple: "
                 << fields.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  tup.setTupleFields(fields.getValue());

  if (!check(TokenKind::ParenClose)) {
    return StringResult<ast::EnumItemTuple>(
        "failed to parse ) token in enum item tuple");
  }
  assert(eat(TokenKind::ParenClose));

  return StringResult<ast::EnumItemTuple>(tup);
}

StringResult<ast::EnumItemStruct> Parser::parseEnumItemStruct() {
  Location loc = getLocation();

  EnumItemStruct str = {loc};

  if (!check(TokenKind::BraceOpen))
    return StringResult<ast::EnumItemStruct>(
        "failed to parse { token in enum item discriminant");

  assert(eat(TokenKind::BraceOpen));

  if (check(TokenKind::BraceClose)) {
    assert(eat(TokenKind::BraceClose));
    return StringResult<ast::EnumItemStruct>(str);
  }

  StringResult<ast::StructFields> fields = parseStructFields();
  if (!fields) {
    llvm::errs() << "failed to parse struct fields in enum item struct: "
                 << fields.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  str.setStructFields(fields.getValue());

  if (!check(TokenKind::BraceClose))
    return StringResult<ast::EnumItemStruct>(
        "failed to parse } token in enum item discriminant");

  assert(eat(TokenKind::BraceClose));

  return StringResult<ast::EnumItemStruct>(str);
}

StringResult<ast::EnumItemDiscriminant> Parser::parseEnumItemDiscriminant() {
  Location loc = getLocation();

  EnumItemDiscriminant dis = {loc};

  if (!check(TokenKind::Eq))
    return StringResult<ast::EnumItemDiscriminant>(
        "failed to parse = token in enum item discriminant");

  assert(eat(TokenKind::Eq));

  Restrictions restrictions;
  StringResult<std::shared_ptr<ast::Expression>> expr = parseExpression({}, restrictions);
  if (!expr) {
    llvm::errs() << "failed to parse expression in enum item discriminant: "
                 << expr.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  dis.setExpression(expr.getValue());

  return StringResult<ast::EnumItemDiscriminant>(dis);
}

StringResult<ast::EnumItems> Parser::parseEnumItems() {
  Location loc = getLocation();

  EnumItems items = {loc};

  StringResult<ast::EnumItem> first = parseEnumItem();
  if (!first) {
    llvm::errs() << "failed to parse enum item in enum items: "
                 << first.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  items.addItem(first.getValue());

  while (true) {
    if (check(TokenKind::Eof)) {
      return StringResult<ast::EnumItems>("failed to parse enum items: eof ");
    } else if (check(TokenKind::BraceClose)) {
      // done
      return StringResult<ast::EnumItems>(items);
    } else if (check(TokenKind::Comma) && check(TokenKind::BraceClose, 1)) {
      assert(eat(TokenKind::Comma));
      // done with trailing
      items.setTrailingComma();
      return StringResult<ast::EnumItems>(items);
    } else if (check(TokenKind::Comma)) {
      assert(eat(TokenKind::Comma));
    } else {
      StringResult<ast::EnumItem> item = parseEnumItem();
      if (!item) {
        llvm::errs() << "failed to parse enum item in enum items: "
                     << item.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      items.addItem(item.getValue());
    }
  }
  return StringResult<ast::EnumItems>("failed to parse enum items");
}

} // namespace rust_compiler::parser
