#include "SelfParam.h"

#include "ShorthandSelf.h"
#include "TypedSelf.h"

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::SelfParam>>
tryParseSelfParam(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  std::optional<std::shared_ptr<ast::SelfParam>> shortSelf =
      tryParseShorthandSelf(view);

  if (shortSelf)
    return *shortSelf;

  std::optional<std::shared_ptr<ast::SelfParam>> typedSelf =
      tryParseTypedSelf(view);

  if (typedSelf)
    return *typedSelf;

  return std::nullopt;
}

} // namespace rust_compiler::parser
