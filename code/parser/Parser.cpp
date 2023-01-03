#include "Parser/Parser.h"

#include "AST/ClippyAttribute.h"
#include "AST/Module.h"
#include "AST/OuterAttribute.h"
#include "Attributes.h"
#include "Lexer/Token.h"
#include "Location.h"
#include "Modules.h"
#include "Visibility.h"

#include <cstdlib>
#include <memory>
#include <optional>

namespace rust_compiler::parser {

using namespace rust_compiler::ast;
using namespace rust_compiler::lexer;

std::shared_ptr<ast::Module> parser(TokenStream &ts,
                                    std::string_view modulePath) {
  Module module = {modulePath};

  std::span<Token> tokens = ts.getAsView();

  size_t last = tokens.size();
  while (tokens.size() > 0) {
    last = tokens.size();

    printf("next token: %zu %s %s %s\n", tokens.size(),
           Token2String(tokens[0].getKind()).c_str(),
           Token2String(tokens[1].getKind()).c_str(),
           Token2String(tokens[2].getKind()).c_str());

    if (tokens.front().getKind() == TokenKind::Hash) {
      if (tokens[1].getKind() == TokenKind::Exclaim) {
        if (tokens[2].getKind() == TokenKind::SquareOpen) {
          std::optional<InnerAttribute> attribute =
              tryParseInnerAttribute(tokens);
          if (attribute) {
            InnerAttribute attr = *attribute;
            tokens = tokens.subspan(attr.getTokens());
            std::shared_ptr<Item> item = std::static_pointer_cast<Item>(
                std::make_shared<InnerAttribute>(attr));
            module.addItem(item);
            continue;
          }
        }
      }
    }

    if (tokens.front().getKind() == TokenKind::Hash) {
      if (tokens[1].getKind() == TokenKind::SquareOpen) {
        std::optional<OuterAttribute> attribute =
            tryParseOuterAttribute(tokens);
        if (attribute) {
          OuterAttribute attr = *attribute;
          tokens = tokens.subspan(attr.getTokens());
          std::shared_ptr<Item> item = std::static_pointer_cast<Item>(
              std::make_shared<OuterAttribute>(attr));
          module.addItem(item);
        }
      }
    }

    if (tokens.front().getKind() == TokenKind::Hash) {
      if (tokens[1].getKind() == TokenKind::Exclaim) {
        if (tokens[2].getKind() == TokenKind::SquareOpen) {
          if (tokens[3].getKind() == TokenKind::Identifier) {
            if (tokens[3].getIdentifier() == "warn" or
                tokens[3].getIdentifier() == "allow" or
                tokens[3].getIdentifier() == "deny") {
              if (tokens[4].getKind() == TokenKind::ParenOpen) {
                std::optional<ClippyAttribute> attribute =
                    tryParseClippyAttribute(tokens);
                if (attribute) {
                  ClippyAttribute attr = *attribute;
                  tokens = tokens.subspan(attr.getTokens());
                  std::shared_ptr<Item> item = std::static_pointer_cast<Item>(
                      std::make_shared<ClippyAttribute>(attr));
                  module.addItem(item);
                }
              }
            }
          }
        }
      }
    }

    if (tokens.front().getKind() == TokenKind::Identifier) {
      if (tokens[1].getIdentifier() == "mod") {
        std::optional<Module> module = tryParseModule(tokens, modulePath);
        if (module) {
          // FIXME
        }
      }
    }

    if (tokens.front().getKind() == TokenKind::Identifier) {
      if (tokens[1].getIdentifier() == "pub") {
        std::optional<Visibility> visibility = tryParseVisibility(tokens);
        if (visibility) {
          tokens = tokens.subspan((*visibility).getTokens());
          std::optional<Module> module = tryParseModule(tokens, modulePath);
          if (module) {
            // FIXME
          }
          // FIXME
        }
      }
    }

    if (tokens.size() == last) {
      printf("parser: no progress\n");
      exit(EXIT_FAILURE);
    }
  }

  return std::make_shared<ast::Module>(module);
}

} // namespace rust_compiler::parser
