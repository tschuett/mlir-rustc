#include "CrateBuilder.h"

#include "ADT/CanonicalPath.h"
#include "AST/Module.h"
#include "CodeGen/DumpLLVMIR.h"
#include "CodeGen/PassPipeLine.h"
#include "CrateBuilder/CrateBuilder.h"
#include "CrateLoader/CrateLoader.h"
#include "Hir/HirDialect.h"
#include "Lexer/Lexer.h"
#include "Mappings/Mappings.h"
#include "ModuleBuilder/ModuleBuilder.h"
#include "ModuleBuilder/Target.h"
#include "Parser/Parser.h"
#include "Sema/Sema.h"

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
#include <mlir/IR/BuiltinOps.h>
#include <sstream>

using namespace rust_compiler::parser;
using namespace rust_compiler::ast;

namespace rust_compiler::rustc {

void buildCrate(std::string_view path, std::string_view crateName,
                basic::Edition edition) {
  // Create `Target`
  std::string theTriple =
      llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple());

  std::string error;
  const llvm::Target *theTarget =
      llvm::TargetRegistry::lookupTarget(theTriple, error);

  std::shared_ptr<ast::Crate> crate =
      crate_loader::loadCrate(path, crateName, edition);

  // llvm::sys::path::append(cargoTomlDir, "src");
  // llvm::sys::path::append(cargoTomlDir, "lib.rs");

  // if (not llvm::sys::fs::exists(cargoTomlDir))
  //   return;

  // llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> outputBuffer =
  //     llvm::MemoryBuffer::getFile(cargoTomlDir, true);
  // if (!outputBuffer) {
  //   return;
  // }
  //
  // std::string str((*outputBuffer)->getBufferStart(),
  //                 (*outputBuffer)->getBufferEnd());
  //
  // lexer::TokenStream ts = lexer::lex(str, "lib.rs");
  // Parser parser = {ts};
  // std::shared_ptr<ast::Module> module = std::make_shared<ast::Module>(
  //     ts.getAsView().front().getLocation(), ast::ModuleKind::Module);
  //(void)parser.parseFile(module);
  //
  // llvm::outs() << "finished parsing: " << module->getItems().size() << "\n";
  //
  std::string fn = "lib.yaml";
  //
  std::error_code EC;
  llvm::raw_fd_stream stream = {fn, EC};

  // std::shared_ptr<Crate> crate = std::make_shared<Crate>(
  //     "toy1", mappings::Mappings::get()->getCrateNum("toy1"));
  // crate->merge(module, adt::CanonicalPath("toy2")); // hack
  //
  // sema::analyzeSemantics(crate);

  llvm::outs() << "code generation"
               << "\n";

  mlir::MLIRContext context;

  rust_compiler::crate_builder::CrateBuilder builder = {stream, context};
  builder.emitCrate(crate);

  mlir::ModuleOp moduleOp = builder.getModule();

  moduleOp.dump();
  // dumpLLVMIR(moduleOp);
  mlir::OwningOpRef<mlir::ModuleOp> owningModuleOp = {moduleOp};
  // processMLIR(context, owningModuleOp);
}

} // namespace rust_compiler::rustc
