#include "AST/Patterns/StructPatternElements.h"
#include "AST/StructExprStruct.h"
#include "AST/StructExprTuple.h"
#include "AST/StructExprUnit.h"
#include "AST/StructFields.h"
#include "AST/StructStruct.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"
#include "Parser/Restrictions.h"

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>

using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace rust_compiler::ast::patterns;
using namespace rust_compiler::lexer;
using namespace llvm;

namespace rust_compiler::parser {

StringResult<ast::StructExprField> Parser::parseStructExprField() {
  Location loc = getLocation();
  StructExprField field = {loc};

  llvm::errs() << "parseStructExprField"
               << "\n";

  StringResult<std::vector<ast::OuterAttribute>> outerAttributes =
      parseOuterAttributes();
  if (!outerAttributes) {
    llvm::errs()
        << "failed to parse outer attributes in parse struct expr field: "
        << outerAttributes.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  std::vector<ast::OuterAttribute> ot = outerAttributes.getValue();
  field.setOuterAttributes(ot);

  if (checkIdentifier() && check(TokenKind::Colon, 1)) {
    field.setIdentifier(getToken().getIdentifier().toString());
    assert(eat(TokenKind::Identifier));
    if (!check(TokenKind::Colon)) {
      return StringResult<ast::StructExprField>(
          "failed to parse : token in struct expr field");
    }
    assert(eat(TokenKind::Colon));
    Restrictions restrictions;
    StringResult<std::shared_ptr<ast::Expression>> expr =
        parseExpression({}, restrictions);
    if (!expr) {
      llvm::errs() << "failed to parse expression in parse struct expr field: "
                   << expr.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    field.setExpression(expr.getValue());
    return StringResult<ast::StructExprField>(field);
  } else if (check(TokenKind::INTEGER_LITERAL) && check(TokenKind::Colon, 1)) {
    field.setTupleIndex(getToken().getLiteral());
    // COPY & PASTE
    assert(eat(TokenKind::INTEGER_LITERAL));
    if (!check(TokenKind::Colon)) {
      return StringResult<ast::StructExprField>(
          "failed to parse : token in struct expr field");
    }
    assert(eat(TokenKind::Colon));
    Restrictions restrictions;
    StringResult<std::shared_ptr<ast::Expression>> expr =
        parseExpression({}, restrictions);
    if (!expr) {
      llvm::errs() << "failed to parse expression in parse struct expr field: "
                   << expr.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    field.setExpression(expr.getValue());
    return StringResult<ast::StructExprField>(field);
  } else if (checkIdentifier()) {
    field.setIdentifier(getToken().getIdentifier().toString());
    return StringResult<ast::StructExprField>(field);
  } else {
    return StringResult<ast::StructExprField>(
        "failed to parse struct expr field");
  }
  return StringResult<ast::StructExprField>(
      "failed to parse struct expr field");
}

StringResult<ast::StructExprFields> Parser::parseStructExprFields() {
  Location loc = getLocation();
  StructExprFields fields = {loc};

  llvm::errs() << "parseStructExprFields"
               << "\n";

  StringResult<StructExprField> first = parseStructExprField();
  if (!first) {
    llvm::errs() << "failed to parse struct expr field in struct expr fields: "
                 << first.getError() << "\n";
    printFunctionStack();
    std::string s =
        llvm::formatv(
            "{0} {1}",
            "failed to parse struct expr field in struct expr fields: ",
            first.getError())
            .str();
    // exit(EXIT_FAILURE);
    return StringResult<ast::StructExprFields>(s);
  }
  fields.addField(first.getValue());

  while (true) {
    if (check(TokenKind::Eof)) {
      return StringResult<ast::StructExprFields>(
          "failed to parse struct expr fields: eof");
    } else if (check(TokenKind::BraceClose)) {
      return StringResult<ast::StructExprFields>(fields);
    } else if (check(TokenKind::Comma) && check(TokenKind::BraceClose, 1)) {
      assert(eat(TokenKind::Comma));
      fields.setTrailingcomma();
      return StringResult<ast::StructExprFields>(fields);
    } else if (check(TokenKind::Comma) && check(TokenKind::DotDot, 1)) {
      assert(eat(TokenKind::Comma));
      StringResult<ast::StructBase> base = parseStructBase();
      if (!base) {
        llvm::errs() << "failed to parse struct base in struct expr fields: "
                     << base.getError() << "\n";
        printFunctionStack();
        std::string s =
            llvm::formatv("{0} {1}",
                          "failed to parse struct base in struct expr fields: ",
                          base.getError())
                .str();
        return StringResult<ast::StructExprFields>(s);
      }
      fields.setBase(base.getValue());
      return StringResult<ast::StructExprFields>(fields);
    } else {
      assert(eat(TokenKind::Comma));
      StringResult<StructExprField> field = parseStructExprField();
      if (!field) {
        llvm::errs()
            << "failed to parse struct expr field in struct expr fields: "
            << field.getError() << "\n";
        printFunctionStack();
        std::string s =
            llvm::formatv(
                "{0} {1}",
                "failed to parse struct expr field in struct expr fields: ",
                field.getError())
                .str();
        return StringResult<ast::StructExprFields>(s);
      }
      fields.addField(field.getValue());
    }
  }
  return StringResult<ast::StructExprFields>(
      "failed to parse struct expr fields");
}

StringResult<ast::StructBase> Parser::parseStructBase() {
  Location loc = getLocation();
  StructBase base = {loc};

  if (!check(TokenKind::DotDot)) {
    return StringResult<ast::StructBase>(
        "failed to parse .. token struct base");
    assert(eat(TokenKind::ParenOpen));
  }

  StringResult<std::shared_ptr<ast::Expression>> path = parsePathInExpression();
  if (!path) {
    llvm::errs() << "failed to parse path in expression in struct base: "
                 << path.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  base.setPath(path.getValue());

  return StringResult<ast::StructBase>(base);
}

StringResult<std::shared_ptr<ast::Expression>> Parser::parseStructExprUnit() {
  Location loc = getLocation();
  StructExprUnit unit = {loc};

  StringResult<std::shared_ptr<ast::Expression>> path = parsePathInExpression();
  if (!path) {
    llvm::errs() << "failed to parse path in expression in struct expr unit: "
                 << path.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  unit.setPath(path.getValue());

  return StringResult<std::shared_ptr<ast::Expression>>(
      std::make_shared<StructExprUnit>(unit));
}

StringResult<std::shared_ptr<ast::Expression>> Parser::parseStructExprTuple() {
  Location loc = getLocation();

  StructExprTuple tuple = {loc};

  StringResult<std::shared_ptr<ast::Expression>> path = parsePathInExpression();
  if (!path) {
    llvm::outs() << "failed to parse path in epxression in struct expr tuple: "
                 << path.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  tuple.setPath(path.getValue());

  if (!check(TokenKind::ParenOpen))
    return StringResult<std::shared_ptr<ast::Expression>>(
        "failed to parse ( token struct expr tuple");
  assert(eat(TokenKind::ParenOpen));

  if (check(TokenKind::ParenClose)) {
    // empty
    assert(eat(TokenKind::ParenClose));
    return StringResult<std::shared_ptr<ast::Expression>>(
        std::make_shared<StructExprTuple>(tuple));
  } else {
    Restrictions restrictions;
    while (true) {
      StringResult<std::shared_ptr<ast::Expression>> expr =
          parseExpression({}, restrictions);
      if (!expr) {
        llvm::outs() << "failed to parse expression in struct expr tuple: "
                     << expr.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      tuple.addExpression(expr.getValue());

      if (check(TokenKind::Eof)) {
        return StringResult<std::shared_ptr<ast::Expression>>(
            "failed to parse struct expr tuple: eof");
      } else if (check(TokenKind::ParenClose)) {
        assert(eat(TokenKind::ParenClose));
        // done
        return StringResult<std::shared_ptr<ast::Expression>>(
            std::make_shared<StructExprTuple>(tuple));
      } else if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
        assert(eat(TokenKind::Comma));
        assert(eat(TokenKind::ParenClose));
        tuple.setTrailingComma();
        // done
        return StringResult<std::shared_ptr<ast::Expression>>(
            std::make_shared<StructExprTuple>(tuple));
      } else if (check(TokenKind::Comma) && !check(TokenKind::ParenClose, 1)) {
        assert(eat(TokenKind::Comma));
        StringResult<std::shared_ptr<ast::Expression>> expr =
            parseExpression({}, restrictions);
        if (!expr) {
          llvm::errs() << "failed to expression in struct expr tuple: "
                       << expr.getError() << "\n";
          printFunctionStack();
          exit(EXIT_FAILURE);
        }
        tuple.addExpression(expr.getValue());
      } else {
        return StringResult<std::shared_ptr<ast::Expression>>(
            "failed to parse struct expr tuple");
      }
    }
  }
  return StringResult<std::shared_ptr<ast::Expression>>(
      "failed to parse struct expr tuple");
}

StringResult<std::shared_ptr<ast::Expression>> Parser::parseStructExprStruct() {
  Location loc = getLocation();

  StructExprStruct str = {loc};

  StringResult<std::shared_ptr<ast::Expression>> path = parsePathInExpression();
  if (!path) {
    llvm::errs() << "failed to parse path in expression in struct expr struct: "
                 << path.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  str.setPath(path.getValue());

  if (!check(TokenKind::BraceOpen))
    return StringResult<std::shared_ptr<ast::Expression>>(
        "failed to parse { token struct expr struct");
  assert(eat(TokenKind::BraceOpen));

  if (check(TokenKind::DotDot)) {
    StringResult<ast::StructBase> base = parseStructBase();
    if (!base) {
      llvm::errs() << "failed to parse struct base in struct expr struct: "
                   << base.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    str.setBase(base.getValue());
  } else {
    StringResult<ast::StructExprFields> fields = parseStructExprFields();
    if (!fields) {
      llvm::errs()
          << "failed to parse struct expr fields in struct expr struct: "
          << fields.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    str.setFields(fields.getValue());
  }

  return StringResult<std::shared_ptr<ast::Expression>>(
      std::make_shared<StructExprStruct>(str));
}

StringResult<std::shared_ptr<ast::Item>>
Parser::parseStructStruct(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();
  class StructStruct str = {loc, vis};

  llvm::errs() << "parseStructStruct"
               << "\n";

  if (!checkKeyWord(KeyWordKind::KW_STRUCT))
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse struct struct");
  assert(eatKeyWord(KeyWordKind::KW_STRUCT));

  if (!checkIdentifier())
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse struct struct");

  str.setName(getToken().getIdentifier());
  assert(eat(TokenKind::Identifier));

  if (check(TokenKind::Lt)) {
    StringResult<ast::GenericParams> params = parseGenericParams();
    if (!params) {
      llvm::errs() << "failed to parse generic params in struct struct: "
                   << params.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    str.setGenericParams(params.getValue());
  }

  if (checkKeyWord(KeyWordKind::KW_WHERE)) {
    StringResult<ast::WhereClause> where = parseWhereClause();
    if (!where) {
      llvm::errs() << "failed to where clause in struct struct: "
                   << where.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    str.setWhereClause(where.getValue());
  }

  if (check(TokenKind::Semi)) {
    assert(eat(TokenKind::Semi));
    return StringResult<std::shared_ptr<ast::Item>>(
        std::make_shared<class StructStruct>(str));
  } else if (check(TokenKind::BraceOpen)) {
    assert(eat(TokenKind::BraceOpen));
    StringResult<ast::StructFields> fields = parseStructFields();
    if (!fields) {
      llvm::errs() << "failed to parse struct fields in struct struct: "
                   << fields.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    str.setFields(fields.getValue());
    if (!check(TokenKind::BraceClose))
      return StringResult<std::shared_ptr<ast::Item>>(
          "failed to parse struct struct");
    assert(eat(TokenKind::BraceClose));
    return StringResult<std::shared_ptr<ast::Item>>(
        std::make_shared<class StructStruct>(str));
  }
  return StringResult<std::shared_ptr<ast::Item>>(
      "failed to parse struct struct");
}

StringResult<std::shared_ptr<ast::Expression>> Parser::parseStructExpression() {

  CheckPoint cp = getCheckPoint();

  if (check(TokenKind::PathSep) || checkPathIdentSegment()) {
    StringResult<std::shared_ptr<ast::Expression>> path =
        parsePathInExpression();
    if (!path) {
      llvm::errs() << "failed to parse path in expession in struct expression: "
                   << path.getError() << "\n";
      printFunctionStack();
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
  } else {
    llvm::outs() << "found no path in parseStructExpression"
                 << "\n";
  }

  return StringResult<std::shared_ptr<ast::Expression>>(
      "failed to parse struct expression");
}

StringResult<ast::StructFields> Parser::parseStructFields() {
  Location loc = getLocation();
  StructFields sfs = {loc};

  StringResult<ast::StructField> sf = parseStructField();
  if (!sf) {
    llvm::errs() << "failed to parse struct field in struct fields: "
                 << sf.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  sfs.addStructField(sf.getValue());

  while (true) {
    if (check(TokenKind::BraceClose)) {
      // done
      return StringResult<ast::StructFields>(sfs);
    } else if (check(TokenKind::Comma) && check(TokenKind::BraceClose, 1)) {
      // done
      assert(eat(TokenKind::Comma));
      sfs.setTrailingComma();
      return StringResult<ast::StructFields>(sfs);
    } else if (check(TokenKind::Eof)) {
      return StringResult<ast::StructFields>("failed to parse struct fields");
    }
    StringResult<ast::StructField> sf = parseStructField();
    if (!sf) {
      llvm::errs() << "failed to parse struct field in struct fields: "
                   << sf.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    sfs.addStructField(sf.getValue());
  }
}

StringResult<ast::StructField> Parser::parseStructField() {
  Location loc = getLocation();
  StructField sf = {loc};

  StringResult<std::vector<ast::OuterAttribute>> outerAttributes =
      parseOuterAttributes();
  if (!outerAttributes) {
    llvm::errs() << "failed to parse outer attributes in struct field: "
                 << outerAttributes.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  std::vector<ast::OuterAttribute> ot = outerAttributes.getValue();
  sf.setOuterAttributes(ot);

  if (checkKeyWord(KeyWordKind::KW_PUB)) {
    StringResult<ast::Visibility> visibility = parseVisibility();
    if (!visibility) {
      llvm::errs() << "failed to parse visibility in struct field: "
                   << visibility.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    sf.setVisibility(visibility.getValue());
  }

  if (!check(TokenKind::Identifier)) {
    return StringResult<ast::StructField>(
        "failed to parse identifier token in struct field");
  }

  Token tok = getToken();
  sf.setIdentifier(tok.getIdentifier());

  assert(eat(TokenKind::Identifier));

  if (!check(TokenKind::Colon)) {
    return StringResult<ast::StructField>(
        "failed to parse : token in struct field");
  }

  assert(eat(TokenKind::Colon));

  StringResult<std::shared_ptr<ast::types::TypeExpression>> type = parseType();
  if (!type) {
    llvm::errs() << "failed to parse type in struct field: " << type.getError()
                 << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  sf.setType(type.getValue());

  return StringResult<ast::StructField>(sf);
}

StringResult<std::shared_ptr<ast::Expression>>
Parser::parseStructExpressionStructPratt(std::shared_ptr<ast::Expression> path,
                                         std::span<ast::OuterAttribute>) {
  StructExprStruct str = {getLocation()};
  str.setPath(path);

  llvm::errs() << "parseStructExpressionStructPratt"
               << "\n";

  if (!check(TokenKind::BraceOpen)) {
    /// error
    return StringResult<std::shared_ptr<ast::Expression>>(
        "failed to parse { token in parseStructExpressionStructPratt");
  }
  assert(eat(TokenKind::BraceOpen));

  if (check(TokenKind::BraceClose)) {
    // done
    assert(eat(TokenKind::BraceClose));
    return StringResult<std::shared_ptr<ast::Expression>>(
        std::make_shared<StructExprStruct>(str));
  }

  if (check(TokenKind::DotDot)) {
    // StructBase
  }

  Result<ast::StructExprFields, std::string> fields = parseStructExprFields();
  if (!fields) {
    // report error
    std::string s =
        llvm::formatv("{0} {1}",
                      "failed to parse struct expr fields in struct "
                      "expression struct pratt",
                      fields.getError())
            .str();
    return StringResult<std::shared_ptr<ast::Expression>>(s);
  }

  str.setFields(fields.getValue());

  if (!check(TokenKind::BraceClose)) {
    /// error
    return StringResult<std::shared_ptr<ast::Expression>>(
        "failed to parse } token in parseStructExpressionStructPratt");
  }
  assert(eat(TokenKind::BraceClose));

  return StringResult<std::shared_ptr<ast::Expression>>(
      std::make_shared<StructExprStruct>(str));
}

adt::StringResult<std::shared_ptr<ast::Expression>>
Parser::parseStructExpressionTuplePratt(std::shared_ptr<ast::Expression> path,
                                        std::span<ast::OuterAttribute> outer) {
  llvm::errs() << "parseStructExpressionTuplePratt"
               << "\n";
  StructExprTuple tuple = {getLocation()};
  tuple.setPath(path);

  if (!check(TokenKind::ParenOpen)) {
    // error
    return StringResult<std::shared_ptr<ast::Expression>>(
        "failed to parse token ( in parse struct expression tuple pratt");
  }
  assert(eat(TokenKind::ParenOpen));

  if (check(TokenKind::ParenClose)) {
    // done: eat
    assert(eat(TokenKind::ParenClose));
    return StringResult<std::shared_ptr<ast::Expression>>(
        std::make_shared<StructExprTuple>(tuple));
  }

  Restrictions restrictions;
  StringResult<std::shared_ptr<ast::Expression>> first =
      parseExpression(outer, restrictions);
  if (!first) {
    // error
    std::string s =
        llvm::formatv("{0} {1}",
                      "failed to parse first expression in struct expr tuple",
                      first.getError())
            .str();
    return StringResult<std::shared_ptr<ast::Expression>>(s);
  }

  tuple.addExpression(first.getValue());

  if (check(TokenKind::ParenClose)) {
    // done: eat
    assert(eat(TokenKind::ParenClose));
    return StringResult<std::shared_ptr<ast::Expression>>(
        std::make_shared<StructExprTuple>(tuple));
  } else if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
    // done: eat
    tuple.setTrailingComma();
    assert(eat(TokenKind::Comma));
    assert(eat(TokenKind::ParenClose));
    return StringResult<std::shared_ptr<ast::Expression>>(
        std::make_shared<StructExprTuple>(tuple));
  } else if (check(TokenKind::Comma) && !check(TokenKind::ParenClose, 1)) {
    assert(eat(TokenKind::Comma));
  }
  while (true) {
    if (check(TokenKind::ParenClose)) {
      // done eat
      assert(eat(TokenKind::ParenClose));
      return StringResult<std::shared_ptr<ast::Expression>>(
          std::make_shared<StructExprTuple>(tuple));
    } else if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
      // done eat
      tuple.setTrailingComma();
      assert(eat(TokenKind::Comma));
      assert(eat(TokenKind::ParenClose));
      return StringResult<std::shared_ptr<ast::Expression>>(
          std::make_shared<StructExprTuple>(tuple));
    } else if (check(TokenKind::Comma) && !check(TokenKind::ParenClose, 1)) {
      assert(eat(TokenKind::Comma));
    } else if (check(TokenKind::Eof)) {
      // error
      std::string s = "failed to parse struct expr tuple: eof";
      return StringResult<std::shared_ptr<ast::Expression>>(s);
    }
    StringResult<std::shared_ptr<ast::Expression>> next =
        parseExpression(outer, restrictions);
    if (!next) {
      // error
      std::string s =
          llvm::formatv("{0} {1}",
                        "failed to parse next expression in struct expr tuple",
                        next.getError())
              .str();
      return StringResult<std::shared_ptr<ast::Expression>>(s);
    }
    tuple.addExpression(next.getValue());
  }

  assert(false && "to be done");
}

} // namespace rust_compiler::parser
