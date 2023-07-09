#inlude "ConstantEvaluation/ConstantEvaluation.h"

namespace rust_compiler::constant_evaluation {

std::optional<Owner> ConstantEvaluation::getOwner(basic::NodeId id) {
  std::optional<Owner> owner = getOwnerCrate(id);

  if (owner) {
    owners[id] = *owner;
    return owner;
  }

  llvm::errs() << "owner failed"
               << "\n";

  return std::nullopt;
  // FIXME caching
}

std::optional<Owner> ConstantEvaluation::getOwner(basic::NodeId id,
                                                  const ast::Crate *crate) {
  assert(id != crate->getNodeId());

  for (auto &item : crate->getItems()) {
    std::optional<Owner> found = getOwnerItem(id, item.get());
    if (found)
      return *found;
  }

  llvm::errs() << "crate failed"
               << "\n";

  return std::nullopt;
}

} // namespace rust_compiler::constant_evaluation
