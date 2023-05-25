#include "AST/PathExprSegment.h"
#include "AST/PathIdentSegment.h"
#include "AST/PathInExpression.h"
#include "AST/QualifiedPathInExpression.h"
#include "AST/SimplePath.h"
#include "AST/SimplePathSegment.h"
#include "AST/Types/QualifiedPathInType.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypePath.h"
#include "AST/Types/TypePathSegment.h"
#include "Lexer/Identifier.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#include <span>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace rust_compiler::ast::types;
using namespace llvm;

namespace rust_compiler::parser {

StringResult<ast::SimplePathSegment> Parser::parseSimplePathSegment() {
  Location loc = getLocation();
  SimplePathSegment seg = {loc};

  if (checkKeyWord(KeyWordKind::KW_SUPER)) {
    seg.setKeyWord(KeyWordKind::KW_SUPER);
    assert(eatKeyWord(KeyWordKind::KW_SUPER));
    return StringResult<ast::SimplePathSegment>(seg);
  } else if (checkKeyWord(KeyWordKind::KW_SELFVALUE)) {
    seg.setKeyWord(KeyWordKind::KW_SELFVALUE);
    assert(eatKeyWord(KeyWordKind::KW_SELFVALUE));
    return StringResult<ast::SimplePathSegment>(seg);
  } else if (checkKeyWord(KeyWordKind::KW_CRATE)) {
    seg.setKeyWord(KeyWordKind::KW_CRATE);
    assert(eatKeyWord(KeyWordKind::KW_CRATE));
    return StringResult<ast::SimplePathSegment>(seg);
  } else if (check(TokenKind::Dollar) &&
             checkKeyWord(KeyWordKind::KW_CRATE, 1)) {
    seg.setIdentifier(Identifier("$crate"));
    assert(eat(TokenKind::Dollar));
    assert(eatKeyWord(KeyWordKind::KW_CRATE));
    return StringResult<ast::SimplePathSegment>(seg);
  } else if (checkIdentifier()) {
    assert(eat(TokenKind::Identifier));
    seg.setIdentifier(getToken().getIdentifier());
    return StringResult<ast::SimplePathSegment>(seg);
  }

  return StringResult<ast::SimplePathSegment>(
      "failed to parse simple path segment");
}

StringResult<std::shared_ptr<ast::Expression>> Parser::parsePathExpression() {
  if (check(TokenKind::Lt))
    return parseQualifiedPathInExpression();

  return parsePathInExpression();
}

StringResult<std::shared_ptr<ast::Expression>>
Parser::parseQualifiedPathInExpression() {
  Location loc = getLocation();

  QualifiedPathInExpression expr = {loc};

  StringResult<ast::types::QualifiedPathType> path = parseQualifiedPathType();
  if (!path) {
    llvm::errs() << "failed to parse qualified pyth type in parse qualified "
                    "path in expression: "
                 << path.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }

  expr.setType(path.getValue());

  while (true) {
    if (check(TokenKind::PathSep)) {
      assert(eat(TokenKind::PathSep));
      StringResult<ast::PathExprSegment> segment = parsePathExprSegment();
      if (!segment) {
        llvm::errs() << "failed to parse path expr segment  in parse qualified "
                        "path in expression: "
                     << segment.getError() << "\n";
        printFunctionStack();
        std::string s =
            llvm::formatv(
                "{0}\n{1}",
                "failed to parse path expr segment  in parse qualified "
                "path in expression: ",
                segment.getError())
                .str();
        return StringResult<std::shared_ptr<ast::Expression>>(s);
      }
      expr.addSegment(segment.getValue());
    } else if (check(TokenKind::Eof)) {
      return StringResult<std::shared_ptr<ast::Expression>>(
          "failed to parse qualified path in expression");
    } else {
      return StringResult<std::shared_ptr<ast::Expression>>(
          std::make_shared<QualifiedPathInExpression>(expr));
    }
  }
  return StringResult<std::shared_ptr<ast::Expression>>(
      "failed to parse qualified path in expression");
}

StringResult<ast::PathExprSegment> Parser::parsePathExprSegment() {
  Location loc = getLocation();
  PathExprSegment seg = {loc};

  //  llvm::errs() << "parsePathExprSegment: " <<
  //  Token2String(getToken().getKind())
  //               << "\n";

  StringResult<ast::PathIdentSegment> first = parsePathIdentSegment();
  if (!first) {
    llvm::errs()
        << "failed to parse path ident segment in parse path expr segment: "
        << first.getError() << "\n";
    printFunctionStack();
    // exit(EXIT_FAILURE);
    std::string s =
        llvm::formatv(
            "{0} {1}",
            "failed to parse path ident segment in parse path expr segment: ",
            first.getError())
            .str();
    return StringResult<ast::PathExprSegment>(s);
  }
  seg.addIdentSegment(first.getValue());

  if (check(TokenKind::PathSep) && check(TokenKind::Lt, 1)) {
    assert(eat(TokenKind::PathSep));
    StringResult<ast::GenericArgs> args = parseGenericArgs();
    if (!args) {
      llvm::errs()
          << "failed to parse generic args in parse path expr segment: "
          << first.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    seg.addGenerics(args.getValue());
  }

  return StringResult<ast::PathExprSegment>(seg);
}

StringResult<std::shared_ptr<ast::Expression>> Parser::parsePathInExpression() {
  Location loc = getLocation();
  PathInExpression path = {loc};

  if (check(TokenKind::PathSep)) {
    path.setLeadingPathSep();
    assert(eat(TokenKind::PathSep));
  }

  StringResult<ast::PathExprSegment> first = parsePathExprSegment();
  if (!first) {
    llvm::errs() << "failed to parse path expr segment in path in expression: "
                 << first.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  path.addSegment(first.getValue());

  while (true) {
    if (check(TokenKind::Eof)) {
      return StringResult<std::shared_ptr<ast::Expression>>(
          std::make_shared<PathInExpression>(path));
    } else if (check(TokenKind::PathSep)) {
      assert(eat(TokenKind::PathSep));
      StringResult<ast::PathExprSegment> next = parsePathExprSegment();
      if (!next) {
        llvm::errs()
            << "failed to parse path expr segment in path in expression: "
            << next.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      path.addSegment(next.getValue());
    } else {
      // done
      return StringResult<std::shared_ptr<ast::Expression>>(
          std::make_shared<PathInExpression>(path));
    }
  }

  return StringResult<std::shared_ptr<ast::Expression>>(
      "failed to parse path in expression");
}

StringResult<ast::PathIdentSegment> Parser::parsePathIdentSegment() {
  Location loc = getLocation();

  PathIdentSegment seg = {loc};

  //  llvm::errs() << "parsePathIdentSegment"
  //               << "\n";

  if (check(TokenKind::Identifier)) {
    Token tok = getToken();
    seg.setIdentifier(tok.getIdentifier());
    // llvm::errs() << "parsePathIdentSegment: " <<
    // tok.getIdentifier().toString() << "\n";
    assert(eat(TokenKind::Identifier));
  } else if (checkKeyWord(KeyWordKind::KW_SUPER)) {
    seg.setSuper();
    assert(eatKeyWord(KeyWordKind::KW_SUPER));
  } else if (checkKeyWord(KeyWordKind::KW_SELFVALUE)) {
    seg.setSelfValue();
    assert(eatKeyWord(KeyWordKind::KW_SELFVALUE));
  } else if (checkKeyWord(KeyWordKind::KW_SELFTYPE)) {
    seg.setSelfType();
    assert(eatKeyWord(KeyWordKind::KW_SELFTYPE));
  } else if (checkKeyWord(KeyWordKind::KW_CRATE)) {
    seg.setCrate();
    assert(eatKeyWord(KeyWordKind::KW_CRATE));
  } else if (check(TokenKind::Dollar) &&
             checkKeyWord(KeyWordKind::KW_CRATE, 1)) {
    seg.setDollarCrate();
    assert(eat(TokenKind::Dollar));
    assert(eatKeyWord(KeyWordKind::KW_CRATE));
  } else {
    llvm::errs() << Token2String(getToken().getKind()) << "\n";
    return StringResult<ast::PathIdentSegment>(
        "failed to parse path ident segment");
  }

  return StringResult<ast::PathIdentSegment>(seg);
}

StringResult<ast::types::TypePathFn> Parser::parseTypePathFn() {
  Location loc = getLocation();

  TypePathFn fn = {loc};

  if (!check(TokenKind::ParenOpen)) {
    return StringResult<ast::types::TypePathFn>(
        "failed to parse ( token in type path fn");
  }
  assert(eat(TokenKind::ParenOpen));

  if (!check(TokenKind::ParenClose)) {
    StringResult<ast::types::TypePathFnInputs> inputs = parseTypePathFnInputs();
    if (!inputs) {
      llvm::errs() << "failed to parse type path fn inputs in type path fn: "
                   << inputs.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    fn.setTypePathFnInputs(inputs.getValue());
  }
  assert(eat(TokenKind::ParenClose));

  if (check(TokenKind::RArrow)) {
    assert(eat(TokenKind::RArrow));
    StringResult<std::shared_ptr<ast::types::TypeExpression>> type =
        parseType();
    if (!type) {
      llvm::errs() << "failed to parse type in type path fn: "
                   << type.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    fn.setType(type.getValue());
  }

  return StringResult<ast::types::TypePathFn>(fn);
}

StringResult<ast::types::TypePathFnInputs> Parser::parseTypePathFnInputs() {
  Location loc = getLocation();

  TypePathFnInputs inputs = {loc};

  StringResult<std::shared_ptr<ast::types::TypeExpression>> start = parseType();
  if (!start) {
    llvm::errs() << "failed to parse type in type path fn inputs: "
                 << start.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  inputs.addType(start.getValue());

  if (!check(TokenKind::Comma)) {
    // done
    return StringResult<ast::types::TypePathFnInputs>(inputs);
  } else if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
    assert(eat(TokenKind::Comma));
    // done
    inputs.setTrailingComma();
    return StringResult<ast::types::TypePathFnInputs>(inputs);
  } else {
    assert(eat(TokenKind::Comma));

    while (true) {
      StringResult<std::shared_ptr<ast::types::TypeExpression>> typ =
          parseType();
      if (!typ) {
        llvm::errs() << "failed to parse type in type path fn inputs: "
                     << typ.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      inputs.addType(typ.getValue());
      if (check(TokenKind::Eof)) {
        // abort
      } else if (check(TokenKind::Comma) && check(TokenKind::ParenClose)) {
        assert(eat(TokenKind::Comma));
        inputs.setTrailingComma();
        return StringResult<ast::types::TypePathFnInputs>(inputs);
      } else if (check(TokenKind::ParenClose)) {
        return StringResult<ast::types::TypePathFnInputs>(inputs);
      } else {
        continue;
      }
    }
  }

  // FIXEM
}

StringResult<ast::types::TypePathSegment> Parser::parseTypePathSegment() {
  Location loc = getLocation();

  TypePathSegment seg = {loc};

  StringResult<ast::PathIdentSegment> ident = parsePathIdentSegment();
  if (!ident) {
    llvm::errs()
        << "failed to parse path ident segment in parse type path segment: "
        << ident.getError() << "\n";
    printFunctionStack();
    // exit(EXIT_FAILURE);
    std::string s =
        llvm::formatv(
            "{0}\n{1}",
            " failed to parse path ident segment in parse type path segment : ",
            ident.getError())
            .str();
    return StringResult<ast::types::TypePathSegment>(s);
  }
  seg.setSegment(ident.getValue());

  // FIXME !!!
  //if (check(TokenKind::PathSep)) {
  //  seg.setDoubleColon();
  //  assert(eat(TokenKind::PathSep));
  //}

  if (check(TokenKind::Lt)) {
    // GenericArgs
    StringResult<ast::GenericArgs> args = parseGenericArgs();
    if (!args) {
      llvm::errs()
          << "failed to parse generic args in parse type path segment: "
          << args.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    seg.setGenericArgs(args.getValue());
    return StringResult<ast::types::TypePathSegment>(seg);
  } else if (check(TokenKind::ParenOpen)) {
    // TypePathFn
    StringResult<ast::types::TypePathFn> fn = parseTypePathFn();
    if (!fn) {
      llvm::errs()
          << "failed to parse type path fn in parse type path segment: "
          << fn.getError() << "\n";
      printFunctionStack();
      // exit(EXIT_FAILURE);
      std::string s =
          llvm::formatv(
              "{0}\n{1}",
              "failed to parse type path fn in parse type path segment: ",
              fn.getError())
              .str();
      return StringResult<ast::types::TypePathSegment>(s);
    }
    seg.setTypePathFn(fn.getValue());
    return StringResult<ast::types::TypePathSegment>(seg);
  }
  return StringResult<ast::types::TypePathSegment>(seg);
}

StringResult<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseTypePath() {
  Location loc = getLocation();

  llvm::errs() << "parseTypePath"
               << "\n";
  llvm::errs() << "parseTypePath: " << Token2String(getToken().getKind())
               << "\n";

  class TypePath path = {loc};

  if (check(TokenKind::PathSep)) {
    path.setLeadingPathSep();
    assert(eat(TokenKind::PathSep));
  }

  StringResult<ast::types::TypePathSegment> first = parseTypePathSegment();
  if (!first) {
    llvm::errs()
        << "failed to first parse type path segment in parse type path: "
        << first.getError() << "\n";
    printFunctionStack();
    // exit(EXIT_FAILURE);
    std::string s =
        llvm::formatv(
            "{0} {1}",
            "failed to first parse type path segment in parse type path: ",
            first.getError())
            .str();
    return StringResult<std::shared_ptr<ast::types::TypeExpression>>(s);
  }
  path.addSegment(first.getValue());

  while (true) {
    llvm::errs() << "parseTypePath: " << Token2String(getToken().getKind())
                 << "\n";
    if (check(TokenKind::Eof)) {
      // error
      return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
          "failed to parse type path (loop): eof");
    } else if (check(TokenKind::PathSep)) {
      assert(eat(TokenKind::PathSep));
      StringResult<ast::types::TypePathSegment> next = parseTypePathSegment();
      if (!next) {
        llvm::errs() << "failed to parse type path segment in parse type path: "
                     << next.getError() << "\n";
        printFunctionStack();
        // exit(EXIT_FAILURE);
        std::string s =
            llvm::formatv(
                "{0} {1}",
                "failed to parse next type path segment in parse type path: ",
                next.getError())
                .str();
        return StringResult<std::shared_ptr<ast::types::TypeExpression>>(s);
      }
      path.addSegment(next.getValue());
    } else {
      return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
          std::make_shared<ast::types::TypePath>(path));
    }
  }

  llvm::errs() << "parseTypePath: " << path.getSegments().size() << "\n";

  return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
      std::make_shared<ast::types::TypePath>(path));
}

StringResult<ast::SimplePath> Parser::parseSimplePath() {
  Location loc = getLocation();

  class SimplePath path {
    loc
  };

  if (check(TokenKind::PathSep)) {
    path.setWithDoubleColon();
    assert(eat(TokenKind::PathSep));
  }

  SimplePathSegment segment = {getLocation()};
  if (check(TokenKind::Identifier)) {
    segment.setIdentifier(getToken().getIdentifier());
  } else if (checkKeyWord(KeyWordKind::KW_SUPER)) {
    segment.setKeyWord(KeyWordKind::KW_SUPER);
  } else if (checkKeyWord(KeyWordKind::KW_SELFVALUE)) {
    segment.setKeyWord(KeyWordKind::KW_SELFVALUE);
  } else if (checkKeyWord(KeyWordKind::KW_CRATE)) {
    segment.setKeyWord(KeyWordKind::KW_CRATE);
  } else if (check(TokenKind::Dollar) &&
             checkKeyWord(KeyWordKind::KW_CRATE, 1)) {
    segment.setIdentifier(Identifier("$crate"));
  } else {
    // error: there are no empty paths
    std::string s =
        llvm::formatv("{0} {1}", "parseSimplePath; empty path: unknown token: ",
                      Token2String(getToken().getKind()))
            .str();
    return StringResult<ast::SimplePath>(s);
  }

  path.addPathSegment(segment);

  if (!check(TokenKind::PathSep)) {
    // done
    return StringResult<ast::SimplePath>(path);
  }
  assert(eat(TokenKind::PathSep));

  while (true) {
    if (check(TokenKind::Eof)) {
      // done
      return StringResult<ast::SimplePath>(path);
    }

    if (check(TokenKind::PathSep)) {
      assert(eat(TokenKind::PathSep));
      SimplePathSegment segment = {getLocation()};

      if (check(TokenKind::Identifier)) {
        segment.setIdentifier(getToken().getIdentifier());
      } else if (checkKeyWord(KeyWordKind::KW_SUPER)) {
        segment.setKeyWord(KeyWordKind::KW_SUPER);
      } else if (checkKeyWord(KeyWordKind::KW_SELFVALUE)) {
        segment.setKeyWord(KeyWordKind::KW_SELFVALUE);
      } else if (checkKeyWord(KeyWordKind::KW_CRATE)) {
        segment.setKeyWord(KeyWordKind::KW_CRATE);
      } else if (check(TokenKind::Dollar) &&
                 checkKeyWord(KeyWordKind::KW_CRATE, 1)) {
        segment.setIdentifier(Identifier("$crate"));
      } else {
        std::string s =
            llvm::formatv("{0} {1}", "parseSimplePath; unknown token: ",
                          Token2String(getToken().getKind()))
                .str();
        return StringResult<ast::SimplePath>(s);
      }
      path.addPathSegment(segment);
    } else {
      // done
      return StringResult<ast::SimplePath>(path);
    }
  }
}

StringResult<ast::types::QualifiedPathType> Parser::parseQualifiedPathType() {
  Location loc = getLocation();
  QualifiedPathType qual = {loc};

  if (!check(TokenKind::Lt))
    return StringResult<ast::types::QualifiedPathType>(
        "failed to parse < token in QualifiedPathType");
  assert(eat(TokenKind::Lt));

  StringResult<std::shared_ptr<ast::types::TypeExpression>> type = parseType();
  if (!type) {
    llvm::errs() << "failed to parse type in parse qualified path type: "
                 << type.getError() << "\n";
    printFunctionStack();
    // exit(EXIT_FAILURE);
    std::string s =
        llvm::formatv("{0}\n{1}",
                      "failed to parse type in parse qualified path type: ",
                      type.getError())
            .str();
    return StringResult<ast::types::QualifiedPathType>(s);
  }
  qual.setType(type.getValue());

  if (checkKeyWord(KeyWordKind::KW_AS)) {
    assert(eatKeyWord(KeyWordKind::KW_AS));
    StringResult<std::shared_ptr<ast::types::TypeExpression>> typePath =
        parseTypePath();
    if (!typePath) {
      llvm::errs() << "failed to parse type path in parse qualified path type: "
                   << typePath.getError() << "\n";
      printFunctionStack();
      std::string s =
          llvm::formatv(
              "{0}\n{1}",
              "failed to parse type path in parse qualified path type: ",
              typePath.getError())
              .str();
      return StringResult<ast::types::QualifiedPathType>(s);
    }
    qual.setPath(typePath.getValue());

    if (!check(TokenKind::Gt))
      return StringResult<ast::types::QualifiedPathType>(
          "failed to parse > token in QualifiedPathType");
    assert(eat(TokenKind::Gt));

    return StringResult<ast::types::QualifiedPathType>(qual);
  }

  if (!check(TokenKind::Gt))
    return StringResult<ast::types::QualifiedPathType>(
        "failed to parse > token in QualifiedPathType");
  assert(eat(TokenKind::Gt));

  return StringResult<ast::types::QualifiedPathType>(qual);
}

StringResult<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseQualifiedPathInType() {
  Location loc = getLocation();
  QualifiedPathInType qual = {loc};

  StringResult<ast::types::QualifiedPathType> seg = parseQualifiedPathType();
  if (!seg) {
    llvm::errs()
        << "failed to parse qualified path type in  qualified path in type: "
        << seg.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  qual.setSegment(seg.getValue());

  if (!check(TokenKind::PathSep)) {
    return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
        std::make_shared<QualifiedPathInType>(qual));
  }

  while (true) {
    if (check(TokenKind::PathSep)) {
      assert(eat(TokenKind::PathSep));
      StringResult<ast::types::TypePathSegment> seg = parseTypePathSegment();
      if (!seg) {
        llvm::errs() << "failed to parse type path segment in parse qualified "
                        "path type: "
                     << seg.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      qual.append(seg.getValue());
    } else if (check(TokenKind::Eof)) {
      return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
          "failed to parse qualified pth in type: eof");
    } else {
      return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
          std::make_shared<QualifiedPathInType>(qual));
    }
  }
  return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
      "failed to parse qualified pth in type");
}

} // namespace rust_compiler::parser
