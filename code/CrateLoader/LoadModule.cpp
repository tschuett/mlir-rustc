#include "LoadModule.h"

#include "Lexer/Lexer.h"
#include "Parser/Parser.h"

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::parser;

namespace rust_compiler::crate_loader {

std::shared_ptr<ast::Module>
loadRootModule(llvm::SmallVectorImpl<char> &libPath, std::string_view crateName,
               adt::CanonicalPath canonicalPath) {

  if (not llvm::sys::fs::exists(libPath)) {
    llvm::outs() << "file: " << libPath << "does not exits" << "\n";
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

  Parser parser = {ts, canonicalPath};

  std::shared_ptr<ast::Module> module = std::make_shared<ast::Module>(
      canonicalPath, ts.getAsView().front().getLocation(),
      ast::ModuleKind::Module);
  if (parser.parseFile(module).succeeded())
    return module;

  llvm::outs() << "parsing failed" << "\n";
  exit(EXIT_FAILURE);
}

} // namespace rust_compiler::crate_loader
