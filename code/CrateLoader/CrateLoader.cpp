#include "CrateLoader/CrateLoader.h"

#include "LoadModule.h"
#include "Sema/Sema.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>

using namespace llvm;
using namespace rust_compiler::adt;

namespace rust_compiler::crate_loader {

std::shared_ptr<ast::Crate> loadCrate(std::string_view path,
                                      std::string_view crateName,
                                      basic::CrateNum crateNum,
                                      basic::Edition edition, LoadMode mode) {

  //  std::shared_ptr<ast::Crate> crate = std::make_shared<ast::Crate>(
  //      crateName, mappings::Mappings::get()->getCrateNum(crateName));

  llvm::SmallVector<char, 128> libFile{path.begin(), path.end()};

  if (mode == LoadMode::SyntaxOnly) {
    if (not llvm::sys::fs::exists(libFile)) {
      llvm::errs() << "could not find file: " << libFile << "\n";
      exit(EXIT_FAILURE);
    }

    std::shared_ptr<ast::Crate> crate =
        loadRootModule(libFile, crateName, crateNum);

    return crate;

  } else if (mode == LoadMode::SyntaxOnly) {
    if (not llvm::sys::fs::exists(libFile)) {
      llvm::errs() << "could not find file: " << libFile << "\n";
      exit(EXIT_FAILURE);
    }

    std::shared_ptr<ast::Crate> crate =
        loadRootModule(libFile, crateName, crateNum);

    sema::analyzeSemantics(crate);

    return crate;

  } else {

    llvm::sys::path::append(libFile, "src");
    llvm::sys::path::append(libFile, "lib.rs");

    if (not llvm::sys::fs::exists(libFile)) {
      llvm::errs() << "could not find file: " << libFile << "\n";
      exit(EXIT_FAILURE);
    }

    std::shared_ptr<ast::Crate> crate =
        loadRootModule(libFile, crateName, crateNum);

    // FIXME load and merge tree

    sema::analyzeSemantics(crate);

    return crate;
  }
}

} // namespace rust_compiler::crate_loader
