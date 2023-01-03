#include "Modules.h"

#include "Lexer/Token.h"

#include <sstream>
#include <string>

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<ast::Module> tryParseModuleTree(std::span<Token> tokens,
                                              std::string_view modulePath) {

  std::span<Token> view = tokens;

  if (view.front().getKind() != TokenKind::Identifier)
    return std::nullopt;

  if (view.front().getKind() == TokenKind::Identifier &&
      view.front().getIdentifier() != "mod")
    return std::nullopt;

  if (view[1].getKind() != TokenKind::Identifier)
    return std::nullopt;

  if (view[2].getKind() != TokenKind::BraceOpen)
    return std::nullopt;

  // everything is fine
}

std::optional<ast::Module> tryParseModule(std::span<Token> tokens,
                                          std::string_view modulePath) {

  std::span<Token> view = tokens;

  if (view.front().getKind() == TokenKind::Identifier &&
      view.front().getIdentifier() == "mod") {
    if (view[1].getKind() == TokenKind::Identifier) {
      if (view[2].getKind() == TokenKind::Colon) {
        std::stringstream s;
        s << modulePath << std::string("::") << view[1].getIdentifier();
        return Module(s.str());
      }
    }
  }

  if (view.front().getKind() == TokenKind::Identifier &&
      view.front().getIdentifier() == "mod") {
    if (view[1].getKind() == TokenKind::Identifier) {
      if (view[2].getKind() == TokenKind::BraceOpen) {
        return tryParseModuleTree(tokens, modulePath);
      }
    }
  }

  return std::nullopt;
}

} // namespace rust_compiler::parser
