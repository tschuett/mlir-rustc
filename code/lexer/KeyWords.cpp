#include "Lexer/KeyWords.h"

namespace rust_compiler::lexer {

// FIXME: slow

static const std::pair<KeyWordKind, std::string> KW[] = {
    {KeyWordKind::KW_AS, "as"},
    {KeyWordKind::KW_BREAK, "break"},
    {KeyWordKind::KW_CONST, "const"},
    {KeyWordKind::KW_CONTINUE, "continue"},
    {KeyWordKind::KW_CRATE, "crate"},
    {KeyWordKind::KW_ELSE, "else"},
    {KeyWordKind::KW_ENUM, "enum"},
    {KeyWordKind::KW_EXTERN, "extern"},
    {KeyWordKind::KW_FALSE, "false"},
    {KeyWordKind::KW_FN, "fn"},
    {KeyWordKind::KW_FOR, "for"},
    {KeyWordKind::KW_IF, "if"},
    {KeyWordKind::KW_IMPL, "impl"},
    {KeyWordKind::KW_IN, "in"},
    {KeyWordKind::KW_LET, "let"},
    {KeyWordKind::KW_LOOP, "loop"},
    {KeyWordKind::KW_MATCH, "match"},
    {KeyWordKind::KW_MOD, "mod"},
    {KeyWordKind::KW_MOVE, "move"},
    {KeyWordKind::KW_MUT, "mut"},
    {KeyWordKind::KW_PUB, "pub"},
    {KeyWordKind::KW_REF, "ref"},
    {KeyWordKind::KW_RETURN, "return"},
    {KeyWordKind::KW_SELFVALUE, "self"},
    {KeyWordKind::KW_SELFTYPE, "Self"},
    {KeyWordKind::KW_STATIC, "static"},
    {KeyWordKind::KW_STRUCT, "struct"},
    {KeyWordKind::KW_SUPER, "super"},
    {KeyWordKind::KW_TRAIT, "trait"},
    {KeyWordKind::KW_TRUE, "true"},
    {KeyWordKind::KW_TYPE, "type"},
    {KeyWordKind::KW_UNSAFE, "unsafe"},
    {KeyWordKind::KW_USE, "use"},
    {KeyWordKind::KW_WHERE, "where"},
    {KeyWordKind::KW_WHILE, "while"},
    {KeyWordKind::KW_ASYNC, "async"},
    {KeyWordKind::KW_AWAIT, "await"},
    {KeyWordKind::KW_DYN, "dyn"},
    {KeyWordKind::KW_ABSTRACT, "abstract"},
    {KeyWordKind::KW_BECOME, "become"},
    {KeyWordKind::KW_BOX, "box"},
    {KeyWordKind::KW_DO, "do"},
    {KeyWordKind::KW_FINAL, "final"},
    {KeyWordKind::KW_MACRO, "macro"},
    {KeyWordKind::KW_OVERRIDE, "override"},
    {KeyWordKind::KW_PRIV, "priv"},
    {KeyWordKind::KW_TYPEOF, "typeof"},
    {KeyWordKind::KW_UNSIZED, "unsized"},
    {KeyWordKind::KW_VIRTUAL, "virtual"},
    {KeyWordKind::KW_YIELD, "yield"},
    {KeyWordKind::KW_TRY, "try"},
    {KeyWordKind::KW_UNION, "union"},
    {KeyWordKind::KW_STATICLIFETIME, "'static"}};

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

} // namespace rust_compiler::lexer
