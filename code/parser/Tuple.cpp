#include "AST/TupleStruct.h"
#include "AST/Types/TupleType.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace rust_compiler::ast::types;
using namespace llvm;

namespace rust_compiler::parser {

StringResult<std::shared_ptr<ast::Item>>
Parser::parseTupleStruct(std::span<OuterAttribute> outer,
                         std::optional<ast::Visibility> vis) {
  Location loc = getLocation();
  class TupleStruct stru = {loc, vis};
  stru.setOuterAttributes(outer);

  if (!checkKeyWord(KeyWordKind::KW_STRUCT))
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse struct keyword in tuple struct");
  assert(eatKeyWord(KeyWordKind::KW_STRUCT));

  if (!checkIdentifier())
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse struct keyword in tuple struct");
  stru.setIdentifier(getToken().getIdentifier());
  assert(eat(TokenKind::Identifier));

  if (check(TokenKind::Lt)) {
    StringResult<ast::GenericParams> generic = parseGenericParams();
    if (!generic) {
      llvm::errs() << "failed to parse generic params in tuple struct: "
                   << generic.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    stru.setGenericParams(generic.getValue());
  }

  if (!check(TokenKind::ParenOpen))
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse ( token in tuple struct");
  assert(eat(TokenKind::ParenOpen));

  if (check(TokenKind::ParenClose)) {
    // FIXME
  } else if (!check(TokenKind::ParenClose)) {
    StringResult<ast::TupleFields> tupleFields = parseTupleFields();
    if (!tupleFields) {
      llvm::errs() << "failed to parse tuple fields in tuple struct: "
                   << tupleFields.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    stru.setTupleFields(tupleFields.getValue());
  }

  if (!check(TokenKind::ParenClose))
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse ) token in tuple struct");
  assert(eat(TokenKind::ParenClose));

  if (checkKeyWord(KeyWordKind::KW_WHERE)) {
    StringResult<ast::WhereClause> where = parseWhereClause();
    if (!where) {
      llvm::errs() << "failed to parse where clause in tuple struct: "
                   << where.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    stru.setWhereClause(where.getValue());
  }

  if (!check(TokenKind::Semi))
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse ; token in tuple struct");
  assert(eat(TokenKind::Semi));

  return StringResult<std::shared_ptr<ast::Item>>(
      std::make_shared<class TupleStruct>(stru));
}

StringResult<ast::TupleField> Parser::parseTupleField() {
  Location loc = getLocation();
  TupleField tuple = {loc};

  if (checkOuterAttribute()) {
    StringResult<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (!outer) {
      llvm::errs() << "failed to parse outer attributes in tuple field: "
                   << outer.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    std::vector<ast::OuterAttribute> ot = outer.getValue();
    tuple.setOuterAttributes(ot);
  }

  if (checkKeyWord(KeyWordKind::KW_PUB)) {
    StringResult<ast::Visibility> vis = parseVisibility();
    if (!vis) {
      llvm::errs() << "failed to parse visibility in tuple field: "
                   << vis.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    tuple.setVisibility(vis.getValue());
  }

  StringResult<std::shared_ptr<ast::types::TypeExpression>> type = parseType();
  if (!type) {
    llvm::errs() << "failed to parse type in tuple field: " << type.getError()
                 << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  tuple.setType(type.getValue());

  return StringResult<ast::TupleField>(tuple);
}

StringResult<ast::TupleFields> Parser::parseTupleFields() {
  Location loc = getLocation();
  TupleFields tuple = {loc};

  StringResult<ast::TupleField> first = parseTupleField();
  if (!first) {
    llvm::errs() << "failed to parse tuple field in tuple fields: "
                 << first.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  tuple.addField(first.getValue());

  while (true) {
    if (check(TokenKind::Eof)) {
      return StringResult<ast::TupleFields>(
          "failed to parse tuple fields: eof");
    } else if (check(TokenKind::ParenClose)) {
      return StringResult<ast::TupleFields>(tuple);
    } else if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
      assert(eat(TokenKind::Comma));
      return StringResult<ast::TupleFields>(tuple);
    } else if (check(TokenKind::Comma)) {
      assert(eat(TokenKind::Comma));
      StringResult<ast::TupleField> next = parseTupleField();
      if (!next) {
        llvm::errs() << "failed to parse tuple field in tuple fields: "
                     << next.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      tuple.addField(next.getValue());
    } else {
      return StringResult<ast::TupleFields>(tuple);
    }
  }
  return StringResult<ast::TupleFields>("failed to parse tuple fields: eof");
}

StringResult<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseTupleType() {
  Location loc = getLocation();

  TupleType tuple = {loc};

  if (check(TokenKind::ParenOpen) && check(TokenKind::ParenClose, 1)) {
    assert(eat(TokenKind::ParenOpen));
    assert(eat(TokenKind::ParenClose));
    return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
        std::make_shared<TupleType>(tuple));
  } else if (!check(TokenKind::ParenOpen)) {
    return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
        "failed to parse ( token in tuple type");
  }

  assert(eat(TokenKind::ParenOpen));

  StringResult<std::shared_ptr<ast::types::TypeExpression>> first = parseType();
  if (!first) {
    llvm::errs() << "failed to parse type in tuple type: " << first.getError()
                 << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  tuple.addType(first.getValue());

  if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
    assert(eat(TokenKind::Comma));
    assert(eat(TokenKind::ParenClose));
    return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
        std::make_shared<TupleType>(tuple));
  } else if (check(TokenKind::ParenClose)) {
    assert(eat(TokenKind::ParenClose));
    return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
        std::make_shared<TupleType>(tuple));
  }

  if (!check(TokenKind::Comma))
    return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
        "failed to parse , token in tuple type");
  assert(eat(TokenKind::Comma));

  while (true) {
    StringResult<std::shared_ptr<ast::types::TypeExpression>> typ = parseType();
    if (!typ) {
      llvm::errs() << "failed to parse type in tuple type: " << typ.getError()
                   << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    tuple.addType(typ.getValue());

    if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
      assert(eat(TokenKind::Comma));
      assert(eat(TokenKind::ParenClose));
      return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
          std::make_shared<TupleType>(tuple));
    } else if (check(TokenKind::Eof)) {
      return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
          "failed to parse tuple type: eof");
    } else if (check(TokenKind::ParenClose)) {
      assert(eat(TokenKind::ParenClose));
      return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
          std::make_shared<TupleType>(tuple));
    } else if (check(TokenKind::Comma)) {
      assert(eat(TokenKind::Comma));
    }
  }
}

} // namespace rust_compiler::parser
