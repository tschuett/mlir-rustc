#pragma once

#include "Frontend/CompilerInvocation.h"

#include <memory>

namespace rust_compiler::frontend {

class CompilerInstance {
  std::unique_ptr<CompilerInvocation> invocation;
};

} // namespace rust_compiler::frontend
