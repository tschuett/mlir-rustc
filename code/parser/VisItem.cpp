#include "Lexer/KeyWords.h"
#include "Parser/Parser.h"

using namespace rust_compiler::ast;
using namespace rust_compiler::lexer;
using namespace llvm;

namespace rust_compiler::parser {

bool Parser::checkVisItem() {
  if (checkKeyWord(KeyWordKind::KW_PUB)) {
    llvm::Expected<ast::Visibility> vis = parseVisibility();
    if (auto e = vis.takeError()) {
      llvm::errs() << "failed to parse visibility in vis item: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
  }

  if (checkKeyWord(KeyWordKind::KW_MOD)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_EXTERN) && checkKeyWord(KeyWordKind::KW_CRATE, 1)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_USE)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_TYPE)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_STRUCT)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_ENUM)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_UNION)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_CONST)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_STATIC)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_TRAIT)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_IMPL)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_UNSAFE) && checkKeyWord(KeyWordKind::KW_IMPL, 1)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_UNSAFE) && checkKeyWord(KeyWordKind::KW_EXTERN, 1)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_EXTERN)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_CONST)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_ASYNC)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_EXTERN)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_FN)) {
    return true;
  }
  return false;
}

} // namespace rust_compiler::parser
