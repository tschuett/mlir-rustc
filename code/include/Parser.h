#pragma once

#include "AST/Module.h"
#include "TokenStream.h"

#include <string_view>

namespace rust_compiler {

std::shared_ptr<ast::Module> parser(TokenStream &ts, std::string_view path);

}
