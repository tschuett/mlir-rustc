#include "Parser.h"

#include "AST/Module.h"
#include "AST/OuterAttribute.h"
#include "Attributes.h"
#include "Visibility.h"

#include <memory>
#include <optional>

namespace rust_compiler {

using namespace rust_compiler::ast;

std::shared_ptr<ast::Module> parser(TokenStream &ts, std::string_view path) {
  Module module = {path};

  std::span<Token> tokens = ts.getAsView();

  while (tokens.size() > 0) {
    if (tokens.front().getKind() == TokenKind::Hash) {
      if (tokens[1].getKind() == TokenKind::Exclaim) {
        if (tokens[2].getKind() == TokenKind::SquareOpen) {
          std::optional<InnerAttribute> attribute =
              tryParseInnerAttribute(tokens);
          tokens = tokens.subspan(attribute->getTokens());
//          std::shared_ptr<Item> item = std::static_pointer_cast<Item>(
//              std::make_shared<OuterAttribute>(attribute));
//          module.addItem(item);
        }
      }
    }
  }

  return std::make_shared<ast::Module>(module);
}

} // namespace rust_compiler
