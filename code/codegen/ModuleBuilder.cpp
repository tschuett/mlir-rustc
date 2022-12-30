#include "ModuleBuilder.h"

#include "Mir/MirDialect.h"
// #include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"

#include <llvm/Remarks/Remark.h>

namespace rust_compiler {

using namespace llvm;
using namespace mlir;

remarks::Remark createRemark(llvm::StringRef pass,
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

Mir::FuncOp ModuleBuilder::buildFun(std::shared_ptr<ast::Function> f) {
  ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);

  // serializer.emit(createRemark("codegen", f->getName()));
  serializer.emit(createRemark("codegen", "fun"));

  builder.setInsertionPointToEnd(theModule.getBody());
  Mir::FuncOp function = buildFunctionSignature(f->getSignature());
  if (!function)
    return nullptr;
}

Mir::FuncOp
ModuleBuilder::buildFunctionSignature(ast::FunctionSignature sig) {
  SmallVector<mlir::Type, 10> argType;

  for (auto &arg : sig.getArgs()) {
    mlir::Type type = getType(arg->getType);
    if (!type)
      return nullptr;

    argType.push_back(type);
  }

  auto funcType = builder.getFunctionType(argTypes, std::nullopt);
  return builder.create<Mir::FuncOp>(location, sig.getName(), funcType);
}

} // namespace rust_compiler
