#include "AST/Crate.h"

#include "TyCtx/TyCtx.h"

namespace rust_compiler::ast {

Crate::Crate(std::string_view crateName, basic::CrateNum crateNum)
    : crateName(crateName), crateNum(crateNum) {
  nodeId = tyctx::TyCtx::get()->getNextNodeId();
};

void Crate::merge(std::shared_ptr<ast::Module> module,
                  adt::CanonicalPath path) {
  assert(false);
}

std::string_view Crate::getCrateName() const { return crateName; }

} // namespace rust_compiler::ast
