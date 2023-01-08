#include "PathIdentSegment.h"

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<std::string>
tryParsePathIdentSegment(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  if (view.front().isIdentifier())
    return view.front().getIdentifier();

  if (view.front().isKeyWord()) {
    std::string key = view.front().getIdentifier();
    if (view.front().getKeyWordKind() == lexer::KeyWordKind::KW_SUPER)
      return std::string("super");
    if (view.front().getKeyWordKind() == lexer::KeyWordKind::KW_SELFVALUE)
      return std::string("self");
    if (view.front().getKeyWordKind() == lexer::KeyWordKind::KW_SELFTYPE)
      return std::string("Self");
    if (view.front().getKeyWordKind() == lexer::KeyWordKind::KW_CRATE)
      return std::string("crate");
  }

  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
