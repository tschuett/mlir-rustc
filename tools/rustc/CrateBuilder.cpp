#include "CrateBuilder.h"

#include "ADT/CanonicalPath.h"
#include "AST/Module.h"
#include "Basic/Ids.h"
#include "CodeGen/DumpLLVMIR.h"
#include "CodeGen/PassPipeLine.h"
#include "CrateBuilder/CrateBuilder.h"
#include "CrateLoader/CrateLoader.h"
#include "Hir/HirDialect.h"
#include "Lexer/Lexer.h"
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
// #include <mlir/Dialect/Arith/IR/Arith.h>
// #include <mlir/Dialect/Async/IR/Async.h>
// #include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
// #include <mlir/Dialect/MemRef/IR/MemRef.h>
// #include <mlir/IR/BuiltinOps.h>
#include <sstream>

using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::crate_loader;

namespace rust_compiler::rustc {

void buildCrate(std::string_view path, std::string_view crateName,
                basic::CrateNum crateNum, basic::Edition edition,
                LoadMode mode) {
  // Create `Target`
  std::string theTriple =
      llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple());

  std::shared_ptr<ast::Crate> crate =
      crate_loader::loadCrate(path, crateName, crateNum, edition, mode);

  std::string fn = "lib.yaml";
  //
  std::error_code EC;
  llvm::raw_fd_stream stream = {fn, EC};

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
