#pragma once

#include "AST/Module.h"
#include "Mir/MirDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include <string_view>

namespace rust_compiler {

class ModuleBuilder {
  std::string moduleName;
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::OpBuilder builder;
  mlir::ModuleOp theModule;

public:
  ModuleBuilder(std::string_view moduleName)
      : moduleName(moduleName), context(), builder(&context) {
    context.getOrLoadDialect<mlir::mir::Mir::MirDialect>();
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  };

  void build(std::shared_ptr<ast::Module> m);

private:
  void buildFun(std::shared_ptr<ast::Function> f);
};

} // namespace rust_compiler
