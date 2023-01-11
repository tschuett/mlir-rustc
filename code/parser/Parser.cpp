#include "Parser/Parser.h"

#include "AST/ClippyAttribute.h"
#include "AST/Module.h"
#include "AST/OuterAttribute.h"
#include "Attributes.h"
#include "Item.h"
#include "Lexer/Token.h"
#include "Location.h"
#include "Modules.h"
#include "Util.h"
#include "Visibility.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <memory>
#include <optional>

namespace rust_compiler::parser {

using namespace rust_compiler::ast;
using namespace rust_compiler::lexer;

//  TokenStream &ts,
//                                            std::string_view modulePath
std::shared_ptr<ast::Module> Parser::parse() {

  std::span<Token> tokens = ts.getAsView();

  Module module = {tokens.front().getLocation(), ModuleKind::Outer, modulePath};

  // ts.print(60);

  size_t last = tokens.size();
  while (tokens.size() > 0) {
    last = tokens.size();

    printTokenState(tokens);

    std::optional<std::shared_ptr<ast::Item>> item =
        tryParseItem(tokens, modulePath);
    if (item) {
      llvm::errs() << "found tokens: " << (*item)->getTokens() << "\n";
      tokens = tokens.subspan((*item)->getTokens());
      module.addItem(*item);
    } else {
      return std::make_shared<ast::Module>(module);
    }

    //    if (tokens.front().getKind() == TokenKind::Hash) {
    //      if (tokens[1].getKind() == TokenKind::Exclaim) {
    //        if (tokens[2].getKind() == TokenKind::SquareOpen) {
    //          std::optional<InnerAttribute> attribute =
    //              tryParseInnerAttribute(tokens);
    //          if (attribute) {
    //            InnerAttribute attr = *attribute;
    //            tokens = tokens.subspan(attr.getTokens());
    //            std::shared_ptr<Item> item = std::static_pointer_cast<Item>(
    //                std::make_shared<InnerAttribute>(attr));
    //            module.addItem(item);
    //            continue;
    //          }
    //        }
    //      }
    //    }
    //
    //    if (tokens.front().getKind() == TokenKind::Hash) {
    //      if (tokens[1].getKind() == TokenKind::SquareOpen) {
    //        std::optional<OuterAttribute> attribute =
    //            tryParseOuterAttribute(tokens);
    //        if (attribute) {
    //          OuterAttribute attr = *attribute;
    //          tokens = tokens.subspan(attr.getTokens());
    //          std::shared_ptr<Item> item = std::static_pointer_cast<Item>(
    //              std::make_shared<OuterAttribute>(attr));
    //          module.addItem(item);
    //        }
    //      }
    //    }
    //
    //    if (tokens.front().getKind() == TokenKind::Hash) {
    //      if (tokens[1].getKind() == TokenKind::Exclaim) {
    //        if (tokens[2].getKind() == TokenKind::SquareOpen) {
    //          if (tokens[3].getKind() == TokenKind::Identifier) {
    //            if (tokens[3].getIdentifier() == "warn" or
    //                tokens[3].getIdentifier() == "allow" or
    //                tokens[3].getIdentifier() == "deny") {
    //              if (tokens[4].getKind() == TokenKind::ParenOpen) {
    //                std::optional<ClippyAttribute> attribute =
    //                    tryParseClippyAttribute(tokens);
    //                if (attribute) {
    //                  ClippyAttribute attr = *attribute;
    //                  tokens = tokens.subspan(attr.getTokens());
    //                  std::shared_ptr<Item> item =
    //                  std::static_pointer_cast<Item>(
    //                      std::make_shared<ClippyAttribute>(attr));
    //                  module.addItem(item);
    //                }
    //              }
    //            }
    //          }
    //        }
    //      }
    //    }
    //
    //    if (tokens.front().getKind() == TokenKind::Identifier) {
    //      if (tokens[1].getIdentifier() == "mod") {
    //        std::optional<Module> moduleOpt = tryParseModule(tokens,
    //        modulePath); if (moduleOpt) {
    //          tokens = tokens.subspan((*moduleOpt).getTokens());
    //          Module mod = *moduleOpt;
    //          std::shared_ptr<Item> modPtr =
    //              std::static_pointer_cast<Item>(std::make_shared<Module>(mod));
    //          module.addItem(modPtr);
    //        }
    //      }
    //    }
    //
    //    if (tokens.front().getKind() == TokenKind::Identifier) {
    //      if (tokens[1].getIdentifier() == "pub") {
    //        std::optional<Visibility> visibility = tryParseVisibility(tokens);
    //        if (visibility) {
    //          tokens = tokens.subspan((*visibility).getTokens());
    //          std::optional<Module> moduleOpt = tryParseModule(tokens,
    //          modulePath); if (moduleOpt) {
    //            tokens = tokens.subspan((*moduleOpt).getTokens());
    //            Module mod = *moduleOpt;
    //            mod.setVisibility(*visibility);
    //            std::shared_ptr<Item> modPtr =
    //                std::static_pointer_cast<Item>(std::make_shared<Module>(mod));
    //            module.addItem(modPtr);
    //          }
    //        } else {
    //          // FIXME
    //        }
    //      }
    //    }

    if (tokens.size() == last) {
      llvm::errs() << "parser: no progress"
                   << "\n";
      printTokenState(tokens);
      exit(EXIT_FAILURE);
    }
  }

  return std::make_shared<ast::Module>(module);
}

} // namespace rust_compiler::parser
