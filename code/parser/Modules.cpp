#include "Modules.h"

#include "AST/Module.h"
#include "Item.h"
#include "Lexer/Token.h"
#include "Util.h"

#include <optional>
#include <sstream>
#include <string>

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<ast::Module> tryParseModuleTree(std::span<Token> tokens,
                                              std::string_view modulePath) {

  Module module = {tokens.front().getLocation(), ModuleKind::ModuleTree,
                   modulePath};

  std::span<Token> view = tokens;

  if (view.front().getKind() != TokenKind::Keyword)
    return std::nullopt;

  if (view.front().getKind() == TokenKind::Keyword &&
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

    //printTokenState(view);

    if (view.front().getKind() == TokenKind::BraceClose) {
      printf("found end of module tree\n");
      return module;
    }

    std::optional<std::shared_ptr<ast::Item>> item =
        tryParseItem(view, modulePath);

    if (item) {
      view = view.subspan((*item)->getTokens());
      module.addItem(*item);
    }

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

  if (view.front().getKind() == TokenKind::Keyword && view.front().getIdentifier() == "mod") {
    if (view[1].getKind() == TokenKind::Identifier) {
      if (view[2].getKind() == TokenKind::SemiColon) {
        std::stringstream s;
        s << modulePath << std::string("::") << view[1].getIdentifier();
        //printf("found module: %s\n", s.str().c_str());
        return Module(view.front().getLocation(), ModuleKind::Module, s.str());
      }
    }
  }

  if (view.front().getKind() == TokenKind::Keyword &&
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
