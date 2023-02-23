#include "AST/Patterns/StructPatternElements.h"
#include "AST/StructExprStruct.h"
#include "AST/StructExprTuple.h"
#include "AST/StructExprUnit.h"
#include "AST/StructStruct.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <memory>

using namespace rust_compiler::ast;
using namespace rust_compiler::ast::patterns;
using namespace rust_compiler::lexer;
using namespace llvm;

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::Expression>> Parser::parseStructExprUnit() {
  Location loc = getLocation();
  StructExprUnit unit = {loc};

  llvm::Expected<std::shared_ptr<ast::PathExpression>> path =
      parsePathInExpression();
  if (auto e = path.takeError()) {
    llvm::errs() << "failed to parse path in expression in struct expr unit : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  unit.setPath(*path);

  return std::make_shared<StructExprUnit>(unit);
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseStructExprTuple() {
  Location loc = getLocation();

  StructExprTuple tuple = {loc};

  llvm::Expected<std::shared_ptr<ast::PathExpression>> path =
      parsePathInExpression();
  if (auto e = path.takeError()) {
    llvm::errs() << "failed to parse path in expression in struct expr tuple : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  tuple.setPath(*path);

  if (!check(TokenKind::ParenOpen))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse ( token struct expr tuple");
  assert(eat(TokenKind::ParenOpen));

  if (check(TokenKind::ParenClose)) {
  } else {
    while (true) {
      llvm::Expected<std::shared_ptr<ast::Expression>> expr = parseExpression();
      if (auto e = expr.takeError()) {
        llvm::errs() << "failed to parse expression in struct expr tuple: "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      tuple.addExpression(*expr);
      if (check(TokenKind::Eof)) {
        return createStringError(inconvertibleErrorCode(),
                                 "failed to parse struct expr tuple: eof");
      } else if (check(TokenKind::ParenClose)) {
        assert(eat(TokenKind::ParenClose));
        // done
        return std::make_shared<StructExprTuple>(tuple);
      } else if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
        assert(eat(TokenKind::Comma));
        assert(eat(TokenKind::ParenClose));
        tuple.setTrailingComma();
        // done
        return std::make_shared<StructExprTuple>(tuple);
      } else if (check(TokenKind::Comma) && !check(TokenKind::ParenClose, 1)) {
        assert(eat(TokenKind::Comma));
        llvm::Expected<std::shared_ptr<ast::Expression>> expr =
            parseExpression();
        if (auto e = expr.takeError()) {
          llvm::errs() << "failed to parse expression in struct expr tuple: "
                       << toString(std::move(e)) << "\n";
          exit(EXIT_FAILURE);
        }
        tuple.addExpression(*expr);
      } else {
        return createStringError(inconvertibleErrorCode(),
                                 "failed to parse struct expr tuple");
      }
    }
  }
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse struct expr tuple");
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseStructExprStruct() {
  Location loc = getLocation();

  StructExprStruct str = {loc};

  llvm::Expected<std::shared_ptr<ast::PathExpression>> path =
      parsePathInExpression();
  if (auto e = path.takeError()) {
    llvm::errs()
        << "failed to parse path in expression in struct expr struct : "
        << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  str.setPath(*path);

  if (!check(TokenKind::BraceOpen))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse { token struct expr struct");
  assert(eat(TokenKind::BraceOpen));

  if (check(TokenKind::DotDot)) {
    llvm::Expected<ast::StructBase> base = parseStructBase();
    if (auto e = base.takeError()) {
      llvm::errs() << "failed to parse struct base in struct expr struct : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    str.setBase(*base);
  } else {
    llvm::Expected<ast::StructExprFields> fields = parseStructExprFields();
    if (auto e = fields.takeError()) {
      llvm::errs()
          << "failed to parse struct expr fields in struct expr struct : "
          << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    str.setFields(*fields);
  }

  return std::make_shared<StructExprStruct>(str);
}

llvm::Expected<std::shared_ptr<ast::VisItem>>
Parser::parseStructStruct(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();
  class StructStruct str = {loc, vis};

  if (!checkKeyWord(KeyWordKind::KW_STRUCT))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse struct struct");
  assert(eatKeyWord(KeyWordKind::KW_STRUCT));

  if (!checkIdentifier())
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse struct struct");

  str.setName(getToken().getIdentifier());
  assert(eat(TokenKind::Identifier));

  if (check(TokenKind::Lt)) {
    llvm::Expected<ast::GenericParams> params = parseGenericParams();
    if (auto e = params.takeError()) {
      llvm::errs() << "failed to parse generic params in struct struct : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    str.setGenericParams(*params);
  }

  if (checkKeyWord(KeyWordKind::KW_STRUCT)) {
    llvm::Expected<ast::WhereClause> where = parseWhereClause();
    if (auto e = where.takeError()) {
      llvm::errs() << "failed to parse where clause in struct struct : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    str.setWhereClause(*where);
  }

  if (check(TokenKind::Semi)) {
    assert(eat(TokenKind::Semi));
    return std::make_shared<class StructStruct>(str);
  } else if (check(TokenKind::BraceOpen)) {
    assert(eat(TokenKind::BraceOpen));
    llvm::Expected<ast::StructFields> fields = parseStructFields();
    if (auto e = fields.takeError()) {
      llvm::errs() << "failed to parse struct fields in struct struct : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    str.setFields(*fields);
    if (!check(TokenKind::BraceClose))
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse struct struct");
    assert(eat(TokenKind::BraceClose));
    return std::make_shared<class StructStruct>(str);
  }
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse struct struct");
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseStructExpression() {

  CheckPoint cp = getCheckPoint();

  if (check(TokenKind::PathSep) || checkPathIdentSegment()) {
    llvm::Expected<std::shared_ptr<ast::PathExpression>> path =
        parsePathInExpression();
    if (auto e = path.takeError()) {
      llvm::errs()
          << "failed to parse path in expression in struct expression : "
          << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }

    if (check(TokenKind::BraceOpen)) {
      recover(cp);
      return parseStructExprStruct();
    } else if (check(TokenKind::ParenOpen)) {
      recover(cp);
      return parseStructExprTuple();
    } else {
      recover(cp);
      return parseStructExprUnit();
    }
  }

  return createStringError(inconvertibleErrorCode(),
                           "failed to parse struct expression");
}

llvm::Expected<ast::StructFields> Parser::parseStructFields() {
  Location loc = getLocation();
  StructFields sfs = {loc};

  llvm::Expected<ast::StructField> sf = parseStructField();
  if (auto e = sf.takeError()) {
    llvm::errs() << "failed to parse struct field in struct fields : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  sfs.addStructField(*sf);

  while (true) {
    if (check(TokenKind::BraceClose)) {
      // done
      return sfs;
    } else if (check(TokenKind::Comma) && check(TokenKind::BraceClose, 1)) {
      // done
      assert(eat(TokenKind::Comma));
      sfs.setTrailingComma();
      return sfs;
    } else if (check(TokenKind::Eof)) {
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse struct fields");
    }
    llvm::Expected<ast::StructField> sf = parseStructField();
    if (auto e = sf.takeError()) {
      llvm::errs() << "failed to parse struct field in struct fields : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    sfs.addStructField(*sf);
  }
}

llvm::Expected<ast::StructField> Parser::parseStructField() {
  Location loc = getLocation();
  StructField sf = {loc};

  llvm::Expected<std::vector<ast::OuterAttribute>> outerAttributes =
      parseOuterAttributes();
  if (auto e = outerAttributes.takeError()) {
    llvm::errs() << "failed to parse outer attributes in struct field : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  sf.setOuterAttributes(*outerAttributes);

  if (checkKeyWord(KeyWordKind::KW_PUB)) {
    llvm::Expected<ast::Visibility> visibility = parseVisibility();
    if (auto e = visibility.takeError()) {
      llvm::errs() << "failed to parse visibility in struct field : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    sf.setVisibility(*visibility);
  }

  if (!check(TokenKind::Identifier)) {
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse identifier token in struct field");
  }

  Token tok = getToken();
  sf.setIdentifier(tok.getIdentifier());

  assert(eat(TokenKind::Identifier));

  if (!check(TokenKind::Colon)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse : token in struct field");
  }

  assert(eat(TokenKind::Colon));

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> type =
      parseType();
  if (auto e = type.takeError()) {
    llvm::errs() << "failed to parse type in struct field : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  sf.setType(*type);

  return sf;
}

} // namespace rust_compiler::parser
