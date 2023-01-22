#include "CrateBuilder.h"

#include "CodeGen/DumpLLVMIR.h"
#include "CodeGen/PassPipeLine.h"
#include "Lexer/Lexer.h"
#include "ModuleBuilder/ModuleBuilder.h"
#include "ModuleBuilder/Target.h"
#include "Parser/Parser.h"
#include "Sema/Sema.h"
#include "mlir/IR/BuiltinOps.h"

#include <fstream>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Host.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <sstream>

using namespace rust_compiler::parser;

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
  cpu = llvm::sys::getHostCPUName();
  std::unique_ptr<::llvm::TargetMachine> tm;
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

  lexer::TokenStream ts = lexer::lex(str, "lib.rs");
  Parser parser = {ts, "crate"};
  std::shared_ptr<ast::Module> module = parser.parse();

  llvm::outs() << "finished parsing: " << module->getItems().size() << "\n";

  sema::analyzeSemantics(module);

  std::string fn = "lib.yaml";

  std::error_code EC;
  llvm::raw_fd_stream stream = {fn, EC};

  llvm::outs() << "code generation"
               << "\n";

  Target target = {tm.get()};
  mlir::MLIRContext context;
  context.getOrLoadDialect<Mir::MirDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::async::AsyncDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();

  rust_compiler::ModuleBuilder mb = {"lib", &target, stream, context};

  mb.build(module);

  mlir::ModuleOp moduleOp = mb.getModule();

  moduleOp.dump();
  // dumpLLVMIR(moduleOp);
  mlir::OwningOpRef<mlir::ModuleOp> owningModuleOp = {moduleOp};
  processMLIR(context, owningModuleOp);
}

} // namespace rust_compiler::rustc
