#include "ModuleBuilder.h"

#include "Mir/MirDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"

namespace rust_compiler {

void ModuleBuilder::build(std::shared_ptr<ast::Module> mod) {
  for (auto f : mod->getFuncs()) {
    buildFun(f);
  }
}

void ModuleBuilder::buildFun(std::shared_ptr<ast::Function> f) {}

} // namespace rust_compiler
