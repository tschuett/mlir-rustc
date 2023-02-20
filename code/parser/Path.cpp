#include "AST/PathIdentSegment.h"
#include "AST/SimplePath.h"
#include "AST/Types/QualifiedPathInType.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypePath.h"
#include "AST/Types/TypePathSegment.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"
#include "llvm/Support/Error.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::ast::types;
using namespace llvm;

namespace rust_compiler::parser {

llvm::Expected<ast::PathIdentSegment> Parser::parsePathIdentSegment() {
  Location loc = getLocation();

  PathIdentSegment seg = {loc};

  if (check(TokenKind::Identifier)) {
    Token tok = getToken();
    seg.setIdentifier(tok.getIdentifier());

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
  } else if (checkKeyWord(KeyWordKind::KW_DOLLARCRATE)) {
    seg.setDollarCrate();
    assert(eatKeyWord(KeyWordKind::KW_DOLLARCRATE));
  } else {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse path ident segment");
  }

  return seg;
}

llvm::Expected<ast::types::TypePathFn> Parser::parseTypePathFn() {
  Location loc = getLocation();

  TypePathFn fn = {loc};

  if (!check(TokenKind::ParenOpen)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse ( token in type path fn");
  }
  assert(eat(TokenKind::ParenOpen));

  if (!check(TokenKind::ParenClose)) {
    llvm::Expected<ast::types::TypePathFnInputs> inputs =
        parseTypePathFnInputs();
    if (auto e = inputs.takeError()) {
      llvm::errs()
          << "failed to parse type path fn inputs in type path path fn: "
          << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    fn.setTypePathFnInputs(*inputs);
  }
  assert(eat(TokenKind::ParenClose));

  if (check(TokenKind::RArrow)) {
    assert(eat(TokenKind::RArrow));
    llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> type =
        parseType();
    if (auto e = type.takeError()) {
      llvm::errs() << "failed to parse type in type path path fn: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    fn.setType(*type);
  }

  return fn;
}

llvm::Expected<ast::types::TypePathFnInputs> Parser::parseTypePathFnInputs() {
  Location loc = getLocation();

  TypePathFnInputs inputs = {loc};

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> start =
      parseType();
  if (auto e = start.takeError()) {
    llvm::errs() << "failed to parse type in type path path fn inputs: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  inputs.addType(*start);

  if (!check(TokenKind::Comma)) {
    // done
    return inputs;
  } else if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
    assert(eat(TokenKind::Comma));
    // done
    inputs.setTrailingComma();
    return inputs;
  } else {
    assert(eat(TokenKind::Comma));

    while (true) {
      llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> typ =
          parseType();
      if (auto e = typ.takeError()) {
        llvm::errs() << "failed to parse type in type path path fn inputs: "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      inputs.addType(*typ);

      if (check(TokenKind::Eof)) {
        // abort
      } else if (check(TokenKind::Comma) && check(TokenKind::ParenClose)) {
        assert(eat(TokenKind::Comma));
        inputs.setTrailingComma();
        return inputs;
      } else if (check(TokenKind::ParenClose)) {
        return inputs;
      } else {
        continue;
      }
    }
  }

  // FIXEM
}

llvm::Expected<ast::types::TypePathSegment> Parser::parseTypePathSegment() {
  Location loc = getLocation();

  TypePathSegment seg = {loc};

  llvm::Expected<ast::PathIdentSegment> ident = parsePathIdentSegment();
  if (auto e = ident.takeError()) {
    llvm::errs() << "failed to parse path ident segment  in type path segment: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  if (check(TokenKind::PathSep)) {
    seg.setDoubleColon();
    assert(eat(TokenKind::PathSep));
  }

  if (check(TokenKind::Lt)) {
    // GenericArgs
    llvm::Expected<ast::GenericArgs> args = parseGenericArgs();
    if (auto e = args.takeError()) {
      llvm::errs() << "failed to parse generic args  in type path segment: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    seg.setGenericArgs(*args);
    return seg;
  } else if (check(TokenKind::ParenOpen)) {
    // TypePathFn
    llvm::Expected<ast::types::TypePathFn> fn = parseTypePathFn();
    if (auto e = fn.takeError()) {
      llvm::errs() << "failed to parse type path fn  in type path segment: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    seg.setTypePathFn(*fn);
    return seg;
  } else {
    return seg;
  }
}

llvm::Expected<std::shared_ptr<ast::types::TypePath>> Parser::parseTypePath() {
  Location loc = getLocation();

  class TypePath path = {loc};

  if (check(TokenKind::PathSep)) {
    path.setLeading();
    assert(eat(TokenKind::PathSep));
  }

  llvm::Expected<ast::types::TypePathSegment> first = parseTypePathSegment();
  if (auto e = first.takeError()) {
    llvm::errs() << "failed to parse type path segment  in type path: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  path.addSegment(*first);

  while (true) {
    if (check(TokenKind::Eof)) {
    } else if (check(TokenKind::PathSep)) {
      assert(eat(TokenKind::PathSep));
      llvm::Expected<ast::types::TypePathSegment> next = parseTypePathSegment();
      if (auto e = next.takeError()) {
        llvm::errs() << "failed to parse type path segment  in type path: "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      path.addSegment(*next);
    } else {
      return std::make_shared<ast::types::TypePath>(path);
    }
  }

  return std::make_shared<ast::types::TypePath>(path);
}

llvm::Expected<ast::SimplePath> Parser::parseSimplePath() {
  Location loc = getLocation();

  class SimplePath path {
    loc
  };

  if (check(TokenKind::PathSep)) {
    path.setWithDoubleColon();
    assert(eat(TokenKind::PathSep));
  }

  if (check(TokenKind::Identifier)) {
  } else if (checkKeyWord(KeyWordKind::KW_SUPER)) {
  } else if (checkKeyWord(KeyWordKind::KW_SELFVALUE)) {
  } else if (checkKeyWord(KeyWordKind::KW_CRATE)) {
  } else if (checkKeyWord(KeyWordKind::KW_DOLLARCRATE)) {
    return path;
  }

  if (!check(TokenKind::PathSep)) {
    // done
  }
  assert(eat(TokenKind::PathSep));

  while (true) {
    if (check(TokenKind::Identifier)) {
    } else if (checkKeyWord(KeyWordKind::KW_SUPER)) {
    } else if (checkKeyWord(KeyWordKind::KW_SELFVALUE)) {
    } else if (checkKeyWord(KeyWordKind::KW_CRATE)) {
    } else if (checkKeyWord(KeyWordKind::KW_DOLLARCRATE)) {
    } else if (check(TokenKind::Eof)) {
    } else {
      return path;
    }

    // PathSep ??
  }
}

llvm::Expected<ast::types::QualifiedPathType> Parser::parseQualifiedPathType() {
  Location loc = getLocation();
  QualifiedPathType qual = {loc};

  if (!check(TokenKind::Lt))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse < token in QualifiedPathType");
  assert(eat(TokenKind::Lt));

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> type =
      parseType();
  if (auto e = type.takeError()) {
    llvm::errs() << "failed to parse  type  in qualified path type: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  qual.setType(*type);

  if (checkKeyWord(KeyWordKind::KW_AS)) {
    assert(eatKeyWord(KeyWordKind::KW_AS));
    llvm::Expected<std::shared_ptr<ast::types::TypePath>> typePath =
        parseTypePath();
    if (auto e = typePath.takeError()) {
      llvm::errs() << "failed to parse  type path  in qualified path type: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    qual.setPath(*typePath);
    assert(eat(TokenKind::Gt));

    return qual;
  }

  return qual;
}

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseQualifiedPathInType() {
  Location loc = getLocation();
  QualifiedPathInType qual = {loc};

  llvm::Expected<ast::types::QualifiedPathType> seg = parseQualifiedPathType();
  if (auto e = seg.takeError()) {
    llvm::errs()
        << "failed to parse qualified path type  in qualified path in type: "
        << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  qual.append(*seg);

  if (!check(TokenKind::PathSep)) {
    return std::make_shared<QualifiedPathInType>(qual);
  }

  while (true) {
    if (check(TokenKind::PathSep)) {
      llvm::Expected<ast::types::QualifiedPathType> seg =
          parseQualifiedPathType();
      if (auto e = seg.takeError()) {
        llvm::errs() << "failed to parse qualified path type  in qualified "
                        "path in type: "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      qual.append(*seg);
    } else if (check(TokenKind::PathSep)) {
      continue;
    } else if (check(TokenKind::Eof)) {
      // abort
    } else {
      return std::make_shared<QualifiedPathInType>(qual);
    }
  }
}

} // namespace rust_compiler::parser
