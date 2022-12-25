#include "CrateBuilder.h"

#include "Lexer.h"
#include "Parser.h"

#include "llvm/Support/MemoryBuffer.h"

#include <fstream>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <sstream>

namespace rust_compiler::minicargo {

void buildCrate(std::string_view path, std::string_view edition) {

  llvm::SmallVector<char, 128> cargoTomlDir{path.begin(), path.end()};

  llvm::sys::path::append(cargoTomlDir, "src");
  llvm::sys::path::append(cargoTomlDir, "lib.rs");

  if (not llvm::sys::fs::exists(cargoTomlDir))
    return;

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> outputBuffer =
      llvm::MemoryBuffer::getFile(cargoTomlDir, true);
  if (!outputBuffer) {
    return;
  }

  std::string str((*outputBuffer)->getBufferStart(),
                  (*outputBuffer)->getBufferEnd());

  //  std::ifstream t(cargoTomlDir);
  //  std::stringstream buffer;
  //  buffer << t.rdbuf();
  //
  //  std::string file = buffer.str();

  TokenStream ts = lex(str);
  parser(ts, "");
}

} // namespace rust_compiler::minicargo
