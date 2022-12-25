#pragma once

#include <optional>
#include <string>

namespace rust_compiler::sema {

/// https://doc.rust-lang.org/reference/keywords.html
enum class KeyWordKind {
  KW_AS,
  KW_BREAK,
  KW_CONST,
  KW_CONTINUE,
  KW_CRATE,
  KW_ELSE,
  KW_ENUM,
  KW_EXTERN,
  KW_FALSE,
  KW_FN,
  KW_FOR,
  KW_IF,
  KW_IMPL,
  KW_IN,
  KW_LET,
  KW_LOOP,
  KW_MATCH,
  KW_MOD,
  KW_MOVE,
  KW_MUT,
  KW_PUB,
  KW_REF,
  KW_RETURN,
  KW_SELFVALUE,
  KW_SELFTYPE,
  KW_STATIC,
  KW_STRUCT,
  KW_SUPER,
  KW_TRAIT,
  KW_TRUE,
  KW_TYPE,
  KW_UNSAFE,
  KW_USE,
  KW_WHERE,
  KW_WHILE,
  KW_ASYNC,
  KW_AWAIT,
  KW_DYN,
  KW_ABSTRACT,
  KW_BECOME,
  KW_BOX,
  KW_DO,
  KW_FINAL,
  KW_MACRO,
  KW_OVERRIDE,
  KW_PRIV,
  KW_TYPEOF,
  KW_UNSIZED,
  KW_VIRTUAL,
  KW_YIELD,
  KW_TRY
};

 extern std::optional<std::string> KeyWord2String(KeyWordKind);

extern std::optional<KeyWordKind> isKeyWord(std::string_view identifier);
 
} // namespace rust_compiler::sema
