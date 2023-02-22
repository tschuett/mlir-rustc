#include "Parser/Parser.h"

using namespace rust_compiler::ast;
using namespace rust_compiler::lexer;
using namespace llvm;

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseStructExpression() {

  CheckPoint cp = getCheckPoint();

  if (checkIntegerLiteral() && check(TokenKind::Colon, 1)) {
    return parseStructExprField();
  } else if (checkIdentifier() && check(TokenKind::Colon, 1)) {
    return parseStructExprField();
  } else if (checkOuterAttribute()) {
    return parseStructExprField();
  } else if (check(TokenKind::PathSep) || checkPathIdentSegment()) {
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
    } else if () {
    }
    xxx
  }
}

  // FIXME only IDENTIFIER in StructExprField
  // FIXME ( in StructExprTuple

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
