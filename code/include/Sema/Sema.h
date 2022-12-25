#pragma once

#include "AST/Module.h"

#include <memory>

namespace rust_compiler::sema {

extern void analyzeSemantics(std::shared_ptr<ast::Module> module);

}
