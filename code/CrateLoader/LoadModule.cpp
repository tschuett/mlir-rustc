#include "LoadModule.h"

#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LogicalResult.h"

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::parser;
using namespace llvm;

namespace rust_compiler::crate_loader {

std::shared_ptr<ast::Crate>
loadRootModule(llvm::SmallVectorImpl<char> &libPath,
               std::string_view crateName) {

  if (not llvm::sys::fs::exists(libPath)) {
    llvm::outs() << "file: " << libPath << "does not exits"
                 << "\n";
    exit(EXIT_FAILURE);
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> outputBuffer =
      llvm::MemoryBuffer::getFile(libPath, true);
  if (!outputBuffer) {
    llvm::outs() << "could not load: " << libPath << "\n";
    exit(EXIT_FAILURE);
  }

  std::string str((*outputBuffer)->getBufferStart(),
                  (*outputBuffer)->getBufferEnd());

  lexer::TokenStream ts = lexer::lex(str, "lib.rs");

  Parser parser = {ts};

  llvm::Expected<std::shared_ptr<ast::Crate>> crate = parser.parseCrateModule(crateName);

  if (auto E = crate.takeError()) {
    errs() << "Failed to parse crate module " << toString(std::move(E)) << "\n";
    exit(EXIT_FAILURE);
  }

  // todo

  return *crate;
}

} // namespace rust_compiler::crate_loader
