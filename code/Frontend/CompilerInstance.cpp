#include "Frontend/CompilerInstance.h"

namespace rust_compiler::frontend {

CompilerInstance::CompilerInstance()
    : invocation(new CompilerInvocation()),
      allSources(new Fortran::parser::AllSources()),
      allCookedSources(new Fortran::parser::AllCookedSources(*allSources)),
      parsing(new Fortran::parser::Parsing(*allCookedSources)) {
  // TODO: This is a good default during development, but ultimately we should
  // give the user the opportunity to specify this.
  allSources->set_encoding(Fortran::parser::Encoding::UTF_8);
}

} // namespace rust_compiler::frontend
