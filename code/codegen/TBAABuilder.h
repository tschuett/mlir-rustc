#pragma once

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinAttributes.h>

namespace rust_compiler::codegen {

class TBAABuilder {
public:
  TBAABuilder(mlir::ModuleOp module);
  TBAABuilder(TBAABuilder const &) = delete;
  TBAABuilder &operator=(TBAABuilder const &) = delete;

  // Attach the llvm.tbaa attribute to the given memory accessing operation
  // based on the provided base/access FIR types and the GEPOp.
//  void attachTBAATag(mlir::Operation *op, mlir::Type baseFIRType,
//                     mlir::Type accessFIRType, mlir::LLVM::GEPOp gep);

private:
  // Return unique string name based on `basename`.
  std::string getNewTBAANodeName(llvm::StringRef basename);

  // LLVM::MetadataOp holding the TBAA operations.
  mlir::LLVM::MetadataOp tbaaMetaOp;
  // Symbol name of tbaaMetaOp.
  static constexpr llvm::StringRef tbaaMetaOpName = "__rustc_tbaa";

  // Base names for TBAA operations:
  //   TBAARootMetadataOp:
  static constexpr llvm::StringRef kRootSymBasename = "root";

  // Symbol defined by the LLVM::TBAARootMetadataOp identifying
  // Rustc's TBAA root.
  mlir::SymbolRefAttr rustcTBAARoot;

  // Identity string for rustc's TBAA root.
  static constexpr llvm::StringRef rustcTBAARootId = "Rustc Type TBAA Root";

  // Counter for unique naming of TBAA operations' symbols.
  unsigned tbaaNodeCounter = 0;
};

} // namespace rust_compiler::codegen
