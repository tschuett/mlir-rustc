#include "Parser/Parser.h"

#include <llvm/Support/raw_os_ostream.h>
#include <optional>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;

namespace rust_compiler::parser {

llvm::Expected<ast::FunctionParameters> Parser::parseFunctionParameters() {

  // FIXME
}

llvm::Expected<ast::FunctionParamPattern> Parser::parseFunctionParamPattern() {
  Location loc = getLocation();

  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>> pattern =
      parsePatternNoTopAlt();

  if (auto e = pattern.takeError()) {
    llvm::errs() << "failed to parse pattern no top alt: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  if (!check(TokenKind::Colon)) {
    // error
  }

  assert(eat(TokenKind::Colon));

  if (check(TokenKind::DotDotDot)) {
    // done
    assert(eat(TokenKind::DotDotDot));
    return FunctionParamPattern(loc, *pattern, /* ... */);
  }

  llvm::Expected<ast::types::TypeExpression> type = parseType();
  if (auto e = type.takeError()) {
    llvm::errs() << "failed to parse type: " << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  // return
}

llvm::Expected<ast::FunctionParam> Parser::parseFunctionParam() {
  std::vector<ast::OuterAttribute> outerAttributes;

  if (checkOuterAttribute()) {
    llvm::Expected<std::vector<ast::OuterAttribute>> parsedOuterAttributes =
        parseOuterAttributes();
    if (auto e = parsedOuterAttributes.takeError()) {
      llvm::errs() << "failed to parse outer attributes: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    outerAttributes = *parsedOuterAttributes;
  }

  if (check(TokenKind::DotDotDot)) {
    // done
    assert(eat(TokenKind::DotDotDot));
    return FunctionParam(loc, *pattern, /* ... */);
  }

  // ???
}

llvm::Expected<std::shared_ptr<ast::BlockExpression>>
Parser::parseFunctionBody() {}

llvm::Expected<ast::FunctionSignature> Parser::parseFunctionsignature() {}

llvm::Expected<ast::FunctionQualifiers> Parser::parseFunctionQualifiers() {}

llvm::Expected<std::shared_ptr<ast::VisItem>>
Parser::parseFunction(std::optional<ast::Visibility> vis) {

  if (checkKeyWord(KeyWordKind::KW_CONST) ||
      checkKeyWord(KeyWordKind::KW_ASYNC) ||
      checkKeyWord(KeyWordKind::KW_UNSAFE) ||
      checkKeyWord(KeyWordKind::KW_EXTERN)) {
    llvm::Expected<ast::FunctionQualifiers> qualifiers =
        parseFunctionQualifiers();
    // check error
  }

  if (!checkKeyWord(KeyWordKind::KW_FN)) {
    // error
  }

  assert(eatKeyWord(KeyWordKind::KW_FN));

  if (!check(TokenKind::Identifier)) {
    // error
  }

  Token id = getToken();
  std::string identifier = id.getIdentifier();

  assert(eat(TokenKind::Identifier));

  if (check(TokenKind::Lt)) {
    llvm::Expected<std::shared_ptr<ast::GenericParams>> genericParams =
        parseGenericParams();
    // check error
  }

  if (!check(TokenKind::ParenOpen)) {
    // error
  }

  assert(eat(TokenKind::ParenOpen));

  // optional
  if (!check(TokenKind::ParenClose)) {
    llvm::Expected<ast::FunctionParameters> parameters =
        parseFunctionParameters();
    // check error
  }

  assert(eat(TokenKind::ParenClose));

  // return type
  if (check(TokenKind::ThinArrow)) {
    assert(eat(TokenKind::ThinArrow));
    llvm::Expected<ast::types::TypeExpression> returnType = parseType();
    // check error
  }

  if (checkKeyWord(KeyWordKind::KW_WHERE)) {
    llvm::Expected<std::shared_ptr<ast::WhereClause>> whereClause =
        parseWhereClause();
    // check error
  }

  if (check(TokenKind::Semi)) {
    assert(eat(TokenKind::Semi));
    // done
  }

  llvm::Expected<std::shared_ptr<ast::BlockExpression>> body =
      parseBlockExpression();
  // check error
}

} // namespace rust_compiler::parser

/*
  TODO:

  https://doc.rust-lang.org/reference/items/generics.html
 */
