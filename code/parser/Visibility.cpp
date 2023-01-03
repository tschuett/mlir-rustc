#include "Visibility.h"

#include "AST/Visiblity.h"
#include "SimplePath.h"

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<Visibility> tryParseVisibility(std::span<Token> tokens) {
  std::span<Token> view = tokens;

  if (view.front().isPubToken()) {
    if (view[1].getKind() != TokenKind::ParenOpen) {
      return Visibility(); // FIXME public
    }
  }

  if (view.front().isPubToken()) {
    if (view[1].getKind() == TokenKind::ParenOpen) {
      if (view[2].isIdentifier()) {
        std::string id = tokens[2].getIdentifier();
        if (id == "crate" or id == "self" or id == "super") {
          return Visibility();
        }

        if (id == "in") {
          view = view.subspan(3); // pub ( in
          std::optional<ast::SimplePath> simplePath = tryParseSimplePath(view);
        }
        // return Visibility();
      }
    }
  }

  return std::nullopt; // FIXME private
}

} // namespace rust_compiler::parser
