#include "Sema/KeyWords.h"

namespace rust_compiler::sema {

// FIXME: slow and incomplete

static const std::pair<KeyWordKind, std::string> KW[] = {
    {KeyWordKind::KW_AS, "abs"},      {KeyWordKind::KW_BREAK, "break"},
    {KeyWordKind::KW_CONST, "const"}, {KeyWordKind::KW_CONTINUE, "continue"},
    {KeyWordKind::KW_CRATE, "crate"}, {KeyWordKind::KW_ELSE, "else"},
    {KeyWordKind::KW_ENUM, "enum"},   {KeyWordKind::KW_EXTERN, "extern"},
    {KeyWordKind::KW_FALSE, "false"},

};

std::optional<std::string> KeyWord2String(KeyWordKind kind) {
  for (auto kw : KW) {
    if (std::get<0>(kw) == kind)
      return std::get<1>(kw);
  }

  return std::nullopt;
}

std::optional<KeyWordKind> isKeyWord(std::string_view identifier) {
  return std::nullopt;
}

} // namespace rust_compiler::sema
