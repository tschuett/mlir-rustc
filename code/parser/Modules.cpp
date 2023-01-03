#include "Modules.h"

#include "Item.h"
#include "Lexer/Token.h"

#include <optional>
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

  view = view.subspan(3);

  size_t last = view.size();

  while (view.size() > 0) {
    last = view.size();

    printf("next token: %zu %s %s %s\n", view.size(),
           Token2String(view[0].getKind()).c_str(),
           Token2String(view[1].getKind()).c_str(),
           Token2String(view[2].getKind()).c_str());

    std::optional<std::shared_ptr<ast::Item>> item =
        tryParseItem(view, modulePath);

    if (view.size() == last) {
      printf("module tree: no progress\n");
      exit(EXIT_FAILURE);
    }
  }

  return std::nullopt;
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
        return Module(view.front().getLocation(), s.str());
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
