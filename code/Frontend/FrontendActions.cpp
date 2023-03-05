#include "Frontend/FrontendActions.h"

#include "Frontend/CompilerInstance.h"

using namespace llvm;

namespace rust_compiler::frontend {

void CodeGenAction::runOptimizationPipeline(llvm::raw_pwrite_stream &os) {
  auto opts = getInstance().getInvocation().getCodeGenOpts();
  auto &diags = getInstance().getDiagnostics();
  llvm::OptimizationLevel level = mapToLevel(opts);

  // Create the analysis managers.
  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  // Create the pass manager builder.
  llvm::PassInstrumentationCallbacks pic;
  llvm::PipelineTuningOptions pto;
  std::optional<llvm::PGOOptions> pgoOpt;
  llvm::StandardInstrumentations si(llvmModule->getContext(),
                                    opts.DebugPassManager);
  si.registerCallbacks(pic, &fam);
  llvm::PassBuilder pb(tm.get(), pto, pgoOpt, &pic);

  // Register all the basic analyses with the managers.
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  // Create the pass manager.
  llvm::ModulePassManager mpm;

  mpm = pb.buildPerModuleDefaultPipeline(level);

  // Run the passes.
  mpm.run(*llvmModule, mam);
}

void CodeGenAction::executeAction() {
  CompilerInstance &ci = this->getInstance();

  if (!llvmModule)
    generateLLVMIR();

  setUpTargetMachine();
  const std::string &theTriple = tm->getTargetTriple().str();

  llvmModule->setTargetTriple(theTriple);
  llvmModule->setDataLayout(tm->createDataLayout());

  runOptimizationPipeline(ci.isOutputStreamNull() ? *os : ci.getOutputStream());

  generateMachineCodeOrAssemblyImpl(
      ci.getDiagnostics(), *tm, action, *llvmModule,
      ci.isOutputStreamNull() ? *os : ci.getOutputStream());
  return;
}

} // namespace rust_compiler::frontend
