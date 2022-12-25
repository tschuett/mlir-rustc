#include "Parser.h"

#include "AST/Module.h"
#include "Attributes.h"
#include "Visibility.h"

#include <optional>

namespace rust_compiler {

using namespace rust_compiler::ast;

void parser(TokenStream &ts, std::string_view path) {

  Module module = {path};

  std::span<Token> tokens = ts.getAsView();

  while (true) {
    if (tokens.front().getKind() == TokenKind::Hash) {
      if (tokens[1].getKind() == TokenKind::SquareOpen) {
        std::optional<OuterAttribute> attribute =
            tryParseOuterAttribute(tokens);
        tokens = tokens.subspan(attribute->getTokens());
      }
    }
  }
}

} // namespace rust_compiler
