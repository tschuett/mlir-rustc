#include "CrateLoader/CrateLoader.h"

#include "LoadModule.h"
#include "Mappings/Mappings.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>

using namespace llvm;
using namespace rust_compiler::adt;
using namespace rust_compiler::mappings;

namespace rust_compiler::crate_loader {

std::shared_ptr<ast::Crate> loadCrate(std::string_view path,
                                      std::string_view crateName,
                                      basic::CrateNum crateNum, LoadMode mode) {
  llvm::SmallVector<char, 128> libFile{path.begin(), path.end()};

  llvm::outs() << "loadCrate" << "\n";
  
  Mappings::get()->setCurrentCrate(crateNum);

  if (mode == LoadMode::File) {
    if (not llvm::sys::fs::exists(libFile)) {
      llvm::errs() << "could not find file: " << libFile << "\n";
      exit(EXIT_FAILURE);
    }

    std::shared_ptr<ast::Crate> crate =
        loadRootModule(libFile, crateName, crateNum);

    return crate;

  } else if (mode == LoadMode::CargoTomlDir) {
    llvm::sys::path::append(libFile, "src");
    llvm::sys::path::append(libFile, "lib.rs");

    if (not llvm::sys::fs::exists(libFile)) {
      llvm::errs() << "could not find file: " << libFile << "\n";
      exit(EXIT_FAILURE);
    }

    std::shared_ptr<ast::Crate> crate =
        loadRootModule(libFile, crateName, crateNum);

    // FIXME load and merge tree

    return crate;
  } else {
    // error
  }
}

} // namespace rust_compiler::crate_loader
