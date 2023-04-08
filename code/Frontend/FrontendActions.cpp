#include "Frontend/FrontendActions.h"

#include "CrateBuilder/CrateBuilder.h"
#include "Frontend/CompilerInstance.h"
#include "Hir/HirDialect.h"
#include "Lir/LirDialect.h"
#include "Mir/MirDialect.h"
#include "Optimizer/PassPipeLine.h"
#include "mlir/IR/BuiltinOps.h"

#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/Triple.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Import.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>

using namespace llvm;
using namespace rust_compiler::optimizer;

/// inspired by Flang

namespace rust_compiler::frontend {

void SyntaxOnlyAction::executeAction() { runParse(); }

void SemaOnlyAction::executeAction() {
  runParse();
  runSemanticChecks();
}

void CodeGenAction::loadDialects(mlir::MLIRContext *context) {
  context->getOrLoadDialect<hir::HirDialect>();
  context->getOrLoadDialect<Mir::MirDialect>();
  context->getOrLoadDialect<Lir::LirDialect>();
  context->getOrLoadDialect<mlir::func::FuncDialect>();
  context->getOrLoadDialect<mlir::arith::ArithDialect>();
  context->getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  context->getOrLoadDialect<mlir::async::AsyncDialect>();
  context->getOrLoadDialect<mlir::memref::MemRefDialect>();
}

void CodeGenAction::setupMLIRModule() {
  mlir::OpBuilder builder(mlirCtx.get());
  mlir::ModuleOp theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  mlirModule = std::make_unique<mlir::ModuleOp>(theModule);
}

void CodeGenAction::setMLIRDataLayout(mlir::ModuleOp &mlirModule,
                                      const llvm::DataLayout &dl) {
  mlir::MLIRContext *context = mlirModule.getContext();
  mlirModule->setAttr(
      mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
      mlir::StringAttr::get(context, dl.getStringRepresentation()));
  //mlir::DataLayoutSpecInterface dlSpec = mlir::translateDataLayout(dl, context);
  //mlirModule->setAttr(mlir::DLTIDialect::kDataLayoutAttrName, dlSpec);
}

/// Runs parsing, sema and lowers to MLIR.
bool CodeGenAction::beginSourceFileAction() {
  llvm::errs() << "beginSourceFileAction"
               << "\n";
  llvmCtx = std::make_unique<llvm::LLVMContext>();
  // CompilerInstance &ci = this->getInstance();

  // Load the MLIR dialects required by rustc
  mlir::DialectRegistry registry;
  mlirCtx = std::make_unique<mlir::MLIRContext>(registry);
  loadDialects(mlirCtx.get());

  bool res = runParse() && runSemanticChecks();
  if (!res)
    return res;

  // Initialize module, so we can set the data layout
  setupMLIRModule();
  setUpTargetMachine();
  const llvm::DataLayout &dl = tm->createDataLayout();
  setMLIRDataLayout(*mlirModule, dl);

  // lower to Hir
  std::error_code EC;
  llvm::raw_fd_ostream OS = {getRemarksOutput(), EC};
  crate_builder::CrateBuilder builder = {OS, *mlirModule.get(), *mlirCtx.get(),
                                         tm.get()};
  builder.emitCrate(getCrate());

  // run the default passes.
  mlir::PassManager pm((*mlirModule)->getName(),
                       mlir::OpPassManager::Nesting::Implicit);
  pm.enableVerifier(/*verifyPasses=*/true);

  if (mlir::failed(pm.run(*mlirModule))) {
    // unsigned diagID = ci.getDiagnostics().getCustomDiagID(
    //     clang::DiagnosticsEngine::Error,
    //     "verification of lowering to FIR failed");
    // ci.getDiagnostics().Report(diagID);
    return false;
  }

  return true;
}

/// Generate target-specific machine-code or assembly file from the input LLVM
/// module.
///
void CodeGenAction::generateObjectFile(llvm::raw_pwrite_stream &os) {
  // Set-up the pass manager, i.e create an LLVM code-gen pass pipeline.
  // Currently only the legacy pass manager is supported.
  // TODO: Switch to the new PM once it's available in the backend.
  llvm::legacy::PassManager codeGenPasses;
  codeGenPasses.add(
      createTargetTransformInfoWrapperPass(tm->getTargetIRAnalysis()));

  llvm::Triple triple(llvmModule->getTargetTriple());
  std::unique_ptr<llvm::TargetLibraryInfoImpl> tlii =
      std::make_unique<llvm::TargetLibraryInfoImpl>(triple);
  assert(tlii && "Failed to create TargetLibraryInfo");
  codeGenPasses.add(new llvm::TargetLibraryInfoWrapperPass(*tlii));

  llvm::CodeGenFileType cgft = llvm::CodeGenFileType::CGFT_ObjectFile;
  if (tm->addPassesToEmitFile(codeGenPasses, os, nullptr, cgft)) {
    // unsigned diagID =
    //     diags.getCustomDiagID(clang::DiagnosticsEngine::Error,
    //                           "emission of this file type is not supported");
    // diags.Report(diagID);
    return;
  }

  // Run the passes
  codeGenPasses.run(*llvmModule);
}

// Lower the previously generated MLIR module into an LLVM IR module
void CodeGenAction::generateLLVMIR() {
  assert(mlirModule && "The MLIR module has not been generated yet.");

  // CompilerInstance &ci = this->getInstance();
  //  auto opts = ci.getInvocation().getCodeGenOpts();
  //  llvm::OptimizationLevel level = llvm::OptimizationLevel::O3;

  loadDialects(mlirCtx.get());

  // fir::support::loadDialects(*mlirCtx);
  // fir::support::registerLLVMTranslation(*mlirCtx);
  mlir::registerLLVMDialectTranslation(*mlirCtx);

  // Set-up the MLIR pass manager
  mlir::PassManager pm((*mlirModule)->getName(),
                       mlir::OpPassManager::Nesting::Implicit);

  // pm.addPass(std::make_unique<Fortran::lower::VerifierPass>());
  pm.enableVerifier(/*verifyPasses=*/true);

  // Create the pass pipeline
  createDefaultOptimizerPassPipeline(pm, ""); // FIXME
  // fir::createMLIRToLLVMPassPipeline(pm, level, opts.StackArrays,
  // opts.Underscoring);
  mlir::applyPassManagerCLOptions(pm);

  // run the pass manager
  if (!mlir::succeeded(pm.run(*mlirModule))) {
    // unsigned diagID = ci.getDiagnostics().getCustomDiagID(
    //     clang::DiagnosticsEngine::Error, "Lowering to LLVM IR failed");
    // ci.getDiagnostics().Report(diagID);
  }

  // Translate to LLVM IR
  std::optional<llvm::StringRef> moduleName = mlirModule->getName();
  llvmModule = mlir::translateModuleToLLVMIR(
      *mlirModule, *llvmCtx, moduleName ? *moduleName : "FIRModule");

  if (!llvmModule) {
    // unsigned diagID = ci.getDiagnostics().getCustomDiagID(
    //     clang::DiagnosticsEngine::Error, "failed to create the LLVM "
    //                                      "module");
    // ci.getDiagnostics().Report(diagID);
    return;
  }

  // Set PIC/PIE level LLVM module flags.
  // if (opts.PICLevel > 0) {
  //  llvmModule->setPICLevel(static_cast<llvm::PICLevel::Level>(opts.PICLevel));
  //  if (opts.IsPIE)
  //    llvmModule->setPIELevel(
  //        static_cast<llvm::PIELevel::Level>(opts.PICLevel));
  //}
}

void CodeGenAction::setUpTargetMachine() {
  // Create `Target`
  std::string theTriple =
      llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple());

  std::string error;
  const llvm::Target *theTarget =
      llvm::TargetRegistry::lookupTarget(theTriple, error);
  assert(theTarget && "Failed to create Target");

  std::string cpu = std::string(llvm::sys::getHostCPUName());

  // Create `TargetMachine`
  tm.reset(theTarget->createTargetMachine(
      theTriple, /*CPU=*/cpu,
      /*Features=*/"", llvm::TargetOptions(),
      /*Reloc::Model=*/Reloc::PIC_,
      /*CodeModel::Model=*/std::nullopt, llvm::CodeGenOpt::Level::Aggressive));
  assert(tm && "Failed to create TargetMachine");
}

void CodeGenAction::runOptimizationPipeline(llvm::raw_pwrite_stream &os) {
  // auto opts = getInstance().getInvocation().getCodeGenOpts();
  // auto &diags = getInstance().getDiagnostics();
  // llvm::OptimizationLevel level = mapToLevel(opts);

  // Create the analysis managers.
  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  // Create the pass manager builder.
  llvm::PassInstrumentationCallbacks pic;
  llvm::PipelineTuningOptions pto;
  std::optional<llvm::PGOOptions> pgoOpt;
  // llvm::StandardInstrumentations si(llvmModule->getContext(),
  // opts.DebugPassManager); si.registerCallbacks(pic, &fam);
  llvm::PassBuilder pb(tm.get(), pto, pgoOpt, &pic);

  // Register all the basic analyses with the
  // managers.
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  // Create the pass manager.
  llvm::ModulePassManager mpm;

  mpm = pb.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);

  // Run the passes.
  mpm.run(*llvmModule, mam);
}

void CodeGenAction::executeAction() {
  if (not beginSourceFileAction())
    return;

  CompilerInstance &ci = this->getInstance();

  if (!llvmModule)
    generateLLVMIR();

  setUpTargetMachine();
  const std::string &theTriple = tm->getTargetTriple().str();

  llvmModule->setTargetTriple(theTriple);
  llvmModule->setDataLayout(tm->createDataLayout());

  std::unique_ptr<llvm::raw_pwrite_stream> output =
      ci.createDefaultOutputFile(getInputFile(), /*extension=*/"o");

  runOptimizationPipeline(*output);

  generateObjectFile(*output);

  return;
}

} // namespace rust_compiler::frontend
