#include "ADT/ScopedCanonicalPath.h"
#include "AST/Module.h"
#include "Item.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"
#include "Util.h"

#include <optional>
#include <sstream>
#include <string>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

namespace rust_compiler::parser {

std::optional<ast::Module>
Parser::tryParseModuleTree(std::span<Token> tokens,
                           std::string_view moduleName) {

  Module module = {path.getCurrentPath().append(moduleName),
                   tokens.front().getLocation(), ModuleKind::ModuleTree};

  std::span<Token> view = tokens;

  if (view.front().getKind() != TokenKind::Keyword)
    return std::nullopt;

  std::optional<ast::Visibility> visibility = tryParseVisibility(tokens);

  if (visibility) {
    // printf("found visibility: %zu\n", (*visibility).getTokens());
    view = view.subspan((*visibility).getTokens());
  }

  if (view.front().getKind() == TokenKind::Keyword &&
      view.front().getIdentifier() != "mod")
    return std::nullopt;

  if (view[1].getKind() != TokenKind::Identifier)
    return std::nullopt;

  if (view[2].getKind() != TokenKind::BraceOpen)
    return std::nullopt;

  // everything is fine

  ScopedCanonicalPathScope scope = {&path, moduleName};

  view = view.subspan(3);

  size_t last = view.size();

  while (view.size() > 0) {
    last = view.size();

    // printTokenState(view);

    if (view.front().getKind() == TokenKind::BraceClose) {
      // printf("found end of module tree\n");
      return module;
    }

    std::optional<std::shared_ptr<ast::Item>> item = tryParseItem(view);

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

std::optional<ast::Module> Parser::tryParseModule(std::span<Token> tokens) {

  std::span<Token> view = tokens;

  if (view.front().getKind() == TokenKind::Keyword &&
      view.front().getIdentifier() == "mod") {
    if (view[1].getKind() == TokenKind::Identifier) {
      if (view[2].getKind() == TokenKind::Semi) {
        return Module(path.getCurrentPath().append(view[1].getIdentifier()),
                      view.front().getLocation(), ModuleKind::Module);
      }
    }
  }

  if (view.front().getKind() == TokenKind::Keyword &&
      view.front().getIdentifier() == "mod") {
    if (view[1].getKind() == TokenKind::Identifier) {
      if (view[2].getKind() == TokenKind::BraceOpen) {
        return tryParseModuleTree(tokens, view[1].getIdentifier());
      }
    }
  }

  return std::nullopt;
}

} // namespace rust_compiler::parser
