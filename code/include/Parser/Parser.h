#pragma once

#include "AST/Module.h"
#include "Lexer/TokenStream.h"

#include <string_view>

namespace rust_compiler::paerser {

  std::shared_ptr<ast::Module> parser(lexer::TokenStream &ts, std::string_view path);

}
