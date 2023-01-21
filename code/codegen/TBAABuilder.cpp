#include "TBAABuilder.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace rust_compiler::codegen {

std::string TBAABuilder::getNewTBAANodeName(llvm::StringRef basename) {
  return (llvm::Twine(basename) + llvm::Twine('_') +
          llvm::Twine(tbaaNodeCounter++))
      .str();
}

TBAABuilder::TBAABuilder(mlir::ModuleOp module) {
  // Create TBAA MetadataOp with the root and basic type descriptors.
  Location loc = module.getLoc();
  MLIRContext *context = module.getContext();
  OpBuilder builder(module.getBody(), module.getBody()->end());
  tbaaMetaOp = builder.create<MetadataOp>(loc, tbaaMetaOpName);
  builder.setInsertionPointToStart(&tbaaMetaOp.getBody().front());

  // Root node.
  auto rootOp = builder.create<TBAARootMetadataOp>(
      loc, getNewTBAANodeName(kRootSymBasename), rustcTBAARootId);
  rustcTBAARoot = FlatSymbolRefAttr::get(rootOp);
}

} // namespace rust_compiler::codegen
