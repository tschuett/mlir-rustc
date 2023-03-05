#include "Frontend/FrontendActions.h"

#include "Frontend/CompilerInstance.h"

namespace rust_compiler::frontend {

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
