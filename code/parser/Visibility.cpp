#include "Visibility.h"

#include "AST/Visiblity.h"
#include "SimplePath.h"

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<Visibility> tryParseVisibility(std::span<Token> tokens) {
  std::span<Token> view = tokens;

  if (view.front().isPubToken()) {
    if (view[1].getKind() != TokenKind::ParenOpen) {
      return Visibility(view.front().getLocation(), VisibilityKind::Public); // FIXME public
    }
  }

  if (view.front().isPubToken()) {
    if (view[1].getKind() == TokenKind::ParenOpen) {
      if (view[2].isIdentifier()) {
        std::string id = tokens[2].getIdentifier();
        if (id == "crate" or id == "self" or id == "super") {
          if (id == "crate")
            return Visibility(view.front().getLocation(), VisibilityKind::PublicCrate);
          if (id == "self")
            return Visibility(view.front().getLocation(), VisibilityKind::PublicSelf);
          if (id == "super")
            return Visibility(view.front().getLocation(), VisibilityKind::PublicSuper);
        } else if (id == "in") {
          view = view.subspan(3); // pub ( in
          std::optional<ast::SimplePath> simplePath = tryParseSimplePath(view);
          if (simplePath) {
            return Visibility(view.front().getLocation(), *simplePath);
            // FIXME mssing ) !!!
          }
        }
        // return Visibility();
      }
    }
  }

  return std::nullopt; // FIXME private
}

} // namespace rust_compiler::parser
