#include "CrateLoader/CrateLoader.h"

#include "LoadModule.h"
#include "Mappings/Mappings.h"
#include "Sema/Sema.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Path.h>

using namespace llvm;
using namespace rust_compiler::adt;

namespace rust_compiler::crate_loader {

std::shared_ptr<ast::Crate> loadCrate(std::string_view path,
                                      std::string_view crateName,
                                      std::string_view edition) {

  std::shared_ptr<ast::Crate> crate = std::make_shared<ast::Crate>(
      crateName, mappings::Mappings::get()->getCrateNum(crateName));

  llvm::SmallVector<char, 128> libFile{path.begin(), path.end()};

  llvm::sys::path::append(libFile, "src");
  llvm::sys::path::append(libFile, "lib.rs");

  std::shared_ptr<ast::Module> rootMod =
      loadRootModule(libFile, crateName, CanonicalPath("crate"));

  // FIXME load and merge tree

  sema::analyzeSemantics(crate);

  return crate;
}

} // namespace rust_compiler::crate_loader
