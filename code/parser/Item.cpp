#include "AST/ConstantItem.h"
#include "AST/Implementation.h"
#include "AST/StaticItem.h"
#include "AST/Struct.h"
#include "AST/Union.h"
#include "Lexer/KeyWords.h"
#include "Parser/Parser.h"

using namespace llvm;
using namespace rust_compiler::ast;
using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::VisItem>>
Parser::parseImplementation(std::optional<ast::Visibility> vis) {
  //Location loc = getLocation();

  //Implementation impl = {loc, vis};

  assert(false);
}

llvm::Expected<std::shared_ptr<ast::VisItem>>
Parser::parseStaticItem(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();

  StaticItem stat = {loc, vis};

  if (!checkKeyWord(KeyWordKind::KW_STATIC))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse static keyword in static item");

  assert(eatKeyWord(KeyWordKind::KW_STATIC));

  if (checkKeyWord(KeyWordKind::KW_MUT)) {
    stat.setMut();
    assert(eatKeyWord(KeyWordKind::KW_MUT));
  }

  if (!check(TokenKind::Identifier))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse identifier in static item");

  Token id = getToken();
  stat.setIdentifier(id.getIdentifier());
  assert(eat(TokenKind::Identifier));

  if (!check(TokenKind::Semi))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse : in static item");
  assert(eat(TokenKind::Semi));

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> typeExpr =
      parseType();
  if (auto e = typeExpr.takeError()) {
    llvm::errs() << "failed to parse type expression in constant item : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  stat.setType(*typeExpr);

  if (check(TokenKind::Semi)) {
    assert(eat(TokenKind::Semi));
    return std::make_shared<StaticItem>(stat);
  } else if (check(TokenKind::Eq)) {
    // initializer
    llvm::Expected<std::shared_ptr<ast::Expression>> init = parseExpression();
    if (auto e = init.takeError()) {
      llvm::errs() << "failed to parse  expression in constant item : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    stat.setInit(*init);
    assert(eat(TokenKind::Semi));
  } else {
    // report error
  }
  return std::make_shared<StaticItem>(stat);
}

llvm::Expected<std::shared_ptr<ast::VisItem>>
Parser::parseConstantItem(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();

  ConstantItem con = {loc, vis};

  if (!checkKeyWord(KeyWordKind::KW_CONST))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse const keyword in constant item");

  assert(eatKeyWord(KeyWordKind::KW_CONST));

  if (check(TokenKind::Underscore)) {
    assert(eat(TokenKind::Underscore));
    con.setIdentifier("_");
  } else if (check(TokenKind::Identifier)) {
    Token id = getToken();
    con.setIdentifier(id.getIdentifier());
    assert(eat(TokenKind::Identifier));
  } else {
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse identifier token in constant item");
  }

  if (!check(TokenKind::Colon)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse colon token in constant item");
  }

  assert(eat(TokenKind::Colon));

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> typeExpr =
      parseType();
  if (auto e = typeExpr.takeError()) {
    llvm::errs() << "failed to parse type expression in constant item : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  con.setType(*typeExpr);

  if (check(TokenKind::Semi)) {
    assert(eat(TokenKind::Semi));
    return std::make_shared<ConstantItem>(con);
  } else if (check(TokenKind::Eq)) {
    assert(eat(TokenKind::Eq));
    // initializer
    llvm::Expected<std::shared_ptr<ast::Expression>> init = parseExpression();
    if (auto e = init.takeError()) {
      llvm::errs() << "failed to parse  expression in constant item : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    con.setInit(*init);
    assert(eat(TokenKind::Semi));
  } else {
    // report error
  }

  return std::make_shared<ConstantItem>(con);
}

llvm::Expected<std::shared_ptr<ast::VisItem>>
Parser::parseUnion(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();

  Union uni = {loc, vis};

  if (!checkKeyWord(KeyWordKind::KW_UNION))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse union keyword in union");

  assert(eatKeyWord(KeyWordKind::KW_UNION));

  if (!check(TokenKind::Identifier))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse identifier token in union");

  assert(check(TokenKind::Identifier));

  if (check(TokenKind::Lt)) {
    llvm::Expected<ast::GenericParams> genericParams = parseGenericParams();
    // check error
  }

  if (checkKeyWord(KeyWordKind::KW_WHERE)) {
    llvm::Expected<ast::WhereClause> whereClasue = parseWhereClause();
    // check error
  }

  if (!check(TokenKind::BraceOpen)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse { token in union");
  }
  assert(check(TokenKind::BraceOpen));

  llvm::Expected<std::shared_ptr<ast::StructFields>> fields =
      parseStructFields();
  // check error

  // return
}

llvm::Expected<std::shared_ptr<ast::VisItem>>
Parser::parseStruct(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();

  // StructStruct stru = {loc, vis};

  if (!checkKeyWord(KeyWordKind::KW_STRUCT))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse struct keyword in struct");

  if (check(TokenKind::Identifier)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse identifier in struct");
  }

  if (check(TokenKind::Lt)) {
    llvm::Expected<ast::GenericParams> genericParams = parseGenericParams();
    // check error
  }

  if (checkKeyWord(KeyWordKind::KW_WHERE)) {
    llvm::Expected<ast::WhereClause> whereClasue = parseWhereClause();
    // check error
  }

  if (check(TokenKind::BraceOpen)) {
    // StructStruct
  }

  if (check(TokenKind::ParenOpen)) {
    // TupleStruct
  }

  // FIXME
}

} // namespace rust_compiler::parser
