#include "AST/VisItem.h"

#include "AST/ClippyAttribute.h"
#include "AST/Function.h"
#include "AST/UseDeclaration.h"
#include "AST/Visiblity.h"
#include "Attributes.h"
#include "Parser/Parser.h"

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::VisItem>>
Parser::tryParseVisItem(std::span<Token> tokens, std::string_view modulePath) {
  std::span<Token> view = tokens;

  std::optional<ast::Visibility> visibility = tryParseVisibility(view);

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
          std::shared_ptr<VisItem> item =
              std::static_pointer_cast<VisItem>(std::make_shared<Module>(mod));
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
          std::shared_ptr<VisItem> item =
              std::static_pointer_cast<VisItem>(std::make_shared<Module>(mod));
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
      return std::static_pointer_cast<VisItem>(
          std::make_shared<UseDeclaration>(use));
    }
  }

  if (view.front().getKind() == TokenKind::Keyword &&
      view.front().getIdentifier() == "fn") {
    std::optional<ast::Function> fun = tryParseFunction(view, modulePath);
    if (fun) {
      if (visibility)
        fun->setVisibility(*visibility);
      std::shared_ptr<VisItem> item =
          std::static_pointer_cast<VisItem>(std::make_shared<ast::Function>(*fun));

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
        std::shared_ptr<VisItem> item = std::static_pointer_cast<VisItem>(
            std::make_shared<ast::Function>(*fun));
        return item;
      }
    }
  }
}

} // namespace rust_compiler::parser
