#include "Function.h"

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<ast::FunctionSignature>
tryParseFunctionSignature(std::span<lexer::Token> tokens) {
  std::span<Token> view = tokens;
  FunctionSignature sig = {view.front().getLocation()};

  if (view.front().getKind() == TokenKind::Keyword) {
    if (view.front().getIdentifier() == "async") {
      sig.setAsync();
      view = view.subspan(1);
    } else if (view.front().getIdentifier() == "const") {
      sig.setConst();
      view = view.subspan(1);
    } else if (view.front().getIdentifier() == "unsafe") {
      sig.setUnsafe();
      view = view.subspan(1);
    }
  };

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
