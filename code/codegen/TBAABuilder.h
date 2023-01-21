#pragma once

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinAttributes.h>

namespace rust_compiler::codegen {

class TBAABuilder {
public:
  TBAABuilder(mlir::ModuleOp module, bool applyTBAA);
  TBAABuilder(TBAABuilder const &) = delete;
  TBAABuilder &operator=(TBAABuilder const &) = delete;

private:
  // LLVM::MetadataOp holding the TBAA operations.
  mlir::LLVM::MetadataOp tbaaMetaOp;
  // Symbol name of tbaaMetaOp.
  static constexpr llvm::StringRef tbaaMetaOpName = "__rustc_tbaa";
};

} // namespace rust_compiler::codegen
