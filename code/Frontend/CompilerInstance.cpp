#include "Frontend/CompilerInstance.h"

namespace rust_compiler::frontend {

CompilerInstance::CompilerInstance() : invocation(new CompilerInvocation()) {}

} // namespace rust_compiler::frontend
