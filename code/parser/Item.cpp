#include "Item.h"

#include "AST/ClippyAttribute.h"
#include "AST/Function.h"
#include "AST/UseDeclaration.h"
#include "AST/Visiblity.h"
#include "Attributes.h"
#include "UseDeclaration.h"
#include "Parser/Parser.h"

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Item>>
Parser::tryParseItem(std::span<Token> tokens, std::string_view modulePath) {

  std::span<Token> view = tokens;

  llvm::errs() << "tryParseItem"
               << "\n";

  if (view.front().getKind() == TokenKind::Hash) {
    if (view[1].getKind() == TokenKind::Not) {
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
                tokens = tokens.subspan(attr.getTokens()); // why?
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

  std::optional<ast::Visibility> visibility = tryParseVisibility(tokens);

  if (visibility) {
    // printf("found visibility: %zu\n", (*visibility).getTokens());
    view = view.subspan((*visibility).getTokens());
  }

  if (view.front().getKind() == TokenKind::Keyword &&
      view.front().getIdentifier() == "mod") {
    if (view[1].getKind() == TokenKind::Identifier) {
      if (view[2].getKind() == TokenKind::Semi) {
        std::optional<Module> module = tryParseModule(view, modulePath);
        if (module) {
          Module mod = *module;
          if (visibility) {
            mod.setVisibility(*visibility);
          }

          tokens = tokens.subspan(mod.getTokens());
          std::shared_ptr<Item> item =
              std::static_pointer_cast<Item>(std::make_shared<Module>(mod));
          return item;
        }
      } else if (view[2].getKind() == TokenKind::BraceOpen) {
        std::optional<Module> module = tryParseModuleTree(view, modulePath);
        if (module) {
          Module mod = *module;
          if (visibility) {
            mod.setVisibility(*visibility);
          }

          tokens = tokens.subspan(mod.getTokens());
          std::shared_ptr<Item> item =
              std::static_pointer_cast<Item>(std::make_shared<Module>(mod));
          return item;
        }
      }
    }
  }

  if (view.front().getKind() == TokenKind::Keyword &&
      view.front().getIdentifier() == "use") {
    std::optional<UseDeclaration> useDeclaration = tryParseUseDeclaration(view);
    if (useDeclaration) {
      // FIXME
      UseDeclaration use = *useDeclaration;
      return std::static_pointer_cast<Item>(
          std::make_shared<UseDeclaration>(use));
    }
  }

  if (view.front().getKind() == TokenKind::Keyword &&
      view.front().getIdentifier() == "fn") {
    std::optional<ast::Function> fun = tryParseFunction(view, modulePath);
    if (fun) {
      if (visibility)
        fun->setVisibility(*visibility);
      std::shared_ptr<Item> item =
          std::static_pointer_cast<Item>(std::make_shared<ast::Function>(*fun));

      return item;
    }
  }

  if (view.front().getKind() == TokenKind::Keyword &&
      (view.front().getIdentifier() == "async" or
       view.front().getIdentifier() == "const")) {
    if (view[1].getKind() == TokenKind::Keyword &&
        view[1].getIdentifier() == "fn") {
      std::optional<ast::Function> fun = tryParseFunction(view, modulePath);
      if (fun) {
        std::shared_ptr<Item> item = std::static_pointer_cast<Item>(
            std::make_shared<ast::Function>(*fun));
        return item;
      }
    }
  }

  return std::nullopt;
}

} // namespace rust_compiler::parser
