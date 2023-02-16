#include "AST/SimplePath.h"
#include "AST/Types/QualifiedPathInType.h"
#include "AST/Types/TypeExpression.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::ast::types;
using namespace llvm;

namespace rust_compiler::parser {

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
