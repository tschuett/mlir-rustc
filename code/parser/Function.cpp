#include "Function.h"

#include "AST/FunctionQualifiers.h"
#include "Generics.h"

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<ast::FunctionQualifiers>
tryParseFunctionQualifiers(std::span<lexer::Token> tokens) {
  std::span<Token> view = tokens;
  FunctionQualifiers qual;

  if (view.front().getKind() == TokenKind::Keyword) {
    if (view.front().getIdentifier() == "async") {
      qual.setAsync();
      view = view.subspan(1);
    } else if (view.front().getIdentifier() == "const") {
      qual.setConst();
      view = view.subspan(1);
    } else if (view.front().getIdentifier() == "unsafe") {
      qual.setUnsafe();
      view = view.subspan(1);
    } else if (view.front().getIdentifier() == "extern") {
      qual.setExtern();
      view = view.subspan(1);
      // FIXME Abi
    } else {
      return std::nullopt;
    }
  }
  return qual;
  // FIXME
}

std::optional<ast::FunctionSignature>
tryParseFunctionSignature(std::span<lexer::Token> tokens) {
  std::span<Token> view = tokens;
  FunctionSignature sig = {view.front().getLocation()};

  std::optional<ast::FunctionQualifiers> qual =
      tryParseFunctionQualifiers(view);

  view = view.subspan(1); // FIXME

  if (view.front().getKind() == TokenKind::Keyword &&
      view.front().getIdentifier() == "fn") {
    view = view.subspan(1);
  } else {
    return std::nullopt;
  }

  if (view.front().getKind() == TokenKind::Identifier) {
    sig.setName(view.front().getIdentifier());
    view = view.subspan(1);
  } else {
    return std::nullopt;
  }

  tryParseGenericParams(view);

  if (view.front().getKind() == TokenKind::ParenOpen) {
    view = view.subspan(1);
  } else {
    return std::nullopt;
  }

  tryParseFunctionParameters(view);

  if (view.front().getKind() == TokenKind::ParenClose) {
    view = view.subspan(1);
  } else {
    return std::nullopt;
  }

  tryParseFunctionReturnType(view);

  tryParseWhereClause(view);

  // BlockExpress or ;

  // todo parse generic params
  // parse parameter
}

std::optional<ast::Function> tryParseFunction(std::span<lexer::Token> tokens,
                                              std::string_view modulePath) {
  std::span<Token> view = tokens;

  std::optional<FunctionSignature> sig = tryParseFunctionSignature(view);

  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser

/*
  TODO:

  https://doc.rust-lang.org/reference/items/generics.html
 */
