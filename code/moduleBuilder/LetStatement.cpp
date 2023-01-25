#include "AST/LetStatementParam.h"
#include "Mir/MirOps.h"
#include "ModuleBuilder/ModuleBuilder.h"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace mlir;
using namespace rust_compiler::ast;

namespace rust_compiler {

void ModuleBuilder::emitLetStatement(
    std::shared_ptr<ast::LetStatement> letStmt) {
  for (LetStatementParam &lit : letStmt->getVarDecls()) {

    mlir::MemRefType mem = mlir::MemRefType::Builder({1}, builder.getI64Type());

    mlir::Value alloca = builder.create<mlir::memref::AllocOp>(
        getLocation(letStmt->getLocation()), mem);

    symbolTable.insert(lit.getName(), {alloca, &lit});

    builder.create<Mir::VarDeclarationOp>(getLocation(letStmt->getLocation()),
                                          builder.getI64Type(), alloca);
  }
}

} // namespace rust_compiler
