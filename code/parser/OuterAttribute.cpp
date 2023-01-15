#include "AST/OuterAttribute.h"

#include "AST/ClippyAttribute.h"
#include "AST/Function.h"
#include "AST/UseDeclaration.h"
#include "AST/Visiblity.h"
#include "Attributes.h"
#include "Item.h"
#include "Parser/Parser.h"

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::OuterAttribute>>
Parser::tryParseOuterAttribute(std::span<Token> tokens,
                               std::string_view modulePath) {

  std::span<Token> view = tokens;

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
                std::shared_ptr<OuterAttribute> outer =
                    std::static_pointer_cast<OuterAttribute>(
                        std::make_shared<ClippyAttribute>(attr));
                return outer;
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

  return std::nullopt;
}

} // namespace rust_compiler::parser
