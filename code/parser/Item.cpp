#include "Item.h"

#include "AST/ClippyAttribute.h"
#include "AST/UseDeclaration.h"
#include "AST/Visiblity.h"
#include "Attributes.h"
#include "UseDeclaration.h"

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Item>>
tryParseItem(std::span<Token> tokens, std::string_view modulePath) {

  std::span<Token> view = tokens;

  std::optional<ast::Visibility> visibility = tryParseVisibility(tokens);

  if (visibility) {
    view = view.subspan((*visibility).getTokens());
  }

  if (view.front().getKind() == TokenKind::Identifier &&
      view.front().getIdentifier() == "mod") {
    if (view[1].getKind() == TokenKind::Identifier) {
      if (view[2].getKind() == TokenKind::SemiColon) {
        std::optional<Module> module = tryParseModule(tokens, modulePath);
        if (module) {
          Module mod = *module;
          tokens = tokens.subspan(mod.getTokens());
          std::shared_ptr<Item> item = std::static_pointer_cast<Item>(
              std::make_shared<Module>(mod));
          return item;
        }
      } else if (view[2].getKind() == TokenKind::BraceOpen) {
        std::optional<Module> module = tryParseModuleTree(tokens, modulePath);
        if (module) {
          // FIXME
        }
      }
    }
  }

  if (view.front().getKind() == TokenKind::Hash) {
    if (view[1].getKind() == TokenKind::Exclaim) {
      if (view[2].getKind() == TokenKind::SquareOpen) {
        if (view[3].getKind() == TokenKind::Identifier) {
          if (view[3].getIdentifier() == "warn" or
              view[3].getIdentifier() == "allow" or
              view[3].getIdentifier() == "deny") {
            if (view[4].getKind() == TokenKind::ParenOpen) {
              std::optional<ClippyAttribute> clippy =
                  tryParseClippyAttribute(view);
              if (clippy) {
                ClippyAttribute attr = *clippy;
                tokens = tokens.subspan(attr.getTokens());
                std::shared_ptr<Item> item = std::static_pointer_cast<Item>(
                    std::make_shared<ClippyAttribute>(attr));
                return item;
              }
            }
          }
        }
      }
      tryParseInnerAttribute(view); // FIXME
    } else {
      tryParseOuterAttribute(view); // FIXME
    }
  }

  if (view.front().getKind() == TokenKind::Identifier &&
      view.front().getIdentifier() == "use") {
    std::optional<UseDeclaration> useDeclaration = tryParseUseDeclaration(view);
  }

  if (view.front().getKind() == TokenKind::Identifier &&
      view.front().getIdentifier() == "fn") {
    tryParseFunction(tokens, modulePath);
  }

  if (view.front().getKind() == TokenKind::Identifier &&
      (view.front().getIdentifier() == "async" or
       view.front().getIdentifier() == "const")) {
    if (view[1].getKind() == TokenKind::Identifier &&
        view[1].getIdentifier() == "fn") {
      tryParseFunction(tokens, modulePath);
    }
  }

  return std::nullopt;
}

} // namespace rust_compiler::parser
