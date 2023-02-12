#pragma once

#include "AST/TypeAlias.h"
#include "AST/UseTree.h"
#include "AST/Visiblity.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"

#include <llvm/Support/Error.h>

/// https://doc.rust-lang.org/nightly/nightly-rustc/rustc_parse/parser/struct.Parser.html#method.new
namespace rust_compiler::parser {

class Parser2 {
public:
  bool check(lexer::TokenKind token);
  bool checkKeyWord(lexer::KeyWordKind keyword);

  bool eatKeyWord(lexer::KeyWordKind keyword);

  llvm::Expected<ast::Visibility> parseVisibility();
  llvm::Expected<ast::use_tree::UseTree> parseUseTree();
  llvm::Expected<ast::TypeAlias> parseTypeAlias();
};

} // namespace rust_compiler::parser
