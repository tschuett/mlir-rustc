#include "ModuleBuilder.h"

#include "Mir/MirDialect.h"
// #include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"

#include <llvm/Remarks/Remark.h>

namespace rust_compiler {

llvm::remarks::Remark createRemark(llvm::StringRef pass,
                                   llvm::StringRef FunctionName) {
  llvm::remarks::Remark r;
  r.PassName = pass;
  r.FunctionName = FunctionName;
  return r;
}

void ModuleBuilder::build(std::shared_ptr<ast::Module> mod) {
  for (auto f : mod->getFuncs()) {
    buildFun(f);
  }
}

void ModuleBuilder::buildFun(std::shared_ptr<ast::Function> f) {
  // serializer.emit();

  //serializer.emit(createRemark("codegen", f->getName()));
  serializer.emit(createRemark("codegen", "fun"));
}

} // namespace rust_compiler
