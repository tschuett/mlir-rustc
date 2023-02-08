#include "CrateBuilder/CrateBuilder.h"

#include <llvm/Support/raw_ostream.h>

namespace rust_compiler::crate_builder {

//void build(std::shared_ptr<rust_compiler::ast::Crate> crate) {
//  mlir::MLIRContext context;
//
//  // FIXME: + .yaml
//  std::string fn = std::string(crate.getCrateName());
//
//  std::error_code EC;
//  llvm::raw_fd_stream stream = {fn, EC};
//
//  CrateBuilder builder = {stream, context};
//
//  builder->emitCrate(crate);
//}

void CrateBuilder::emitCrate(std::shared_ptr<rust_compiler::ast::Crate> crate) {

  for (auto &i : crate->getItems()) {
    emitItem(i);
  }
}

} // namespace rust_compiler::crate_builder
