#include "CrateBuilder.h"

#include "Lexer/Lexer.h"
#include "ModuleBuilder.h"
#include "Parser/Parser.h"
#include "Sema/Sema.h"
#include "Target.h"

#include <fstream>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <sstream>

namespace rust_compiler::rustc {

void buildCrate(std::string_view path, std::string_view edition) {
  // Create `Target`
  std::string theTriple =
      llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple());

  std::string error;
  const llvm::Target *theTarget =
      llvm::TargetRegistry::lookupTarget(theTriple, error);

  std::string featuresStr;
  std::string cpu = "sapphirerapids";
  std::unique_ptr<llvm::TargetMachine> tm;
  tm.reset(theTarget->createTargetMachine(
      theTriple, /*CPU=*/cpu,
      /*Features=*/featuresStr, llvm::TargetOptions(),
      /*Reloc::Model=*/llvm::Reloc::Model::PIC_,
      /*CodeModel::Model=*/std::nullopt, llvm::CodeGenOpt::Aggressive));
  assert(tm && "Failed to create TargetMachine");

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

  lexer::TokenStream ts = lexer::lex(str, "lib.rs");
  std::shared_ptr<ast::Module> module = parser::parser(ts, "");

  sema::analyzeSemantics(module);

  std::string fn = "lib.yaml";

  std::error_code EC;
  llvm::raw_fd_stream stream = {fn, EC};

  rust_compiler::ModuleBuilder mb = {"lib", stream};

  Target target = {tm.get()};
  mb.build(module, target);
}

} // namespace rust_compiler::rustc
