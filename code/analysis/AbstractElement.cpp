#include "Analysis/Attributer/AbstractElement.h"

namespace rust_compiler::analysis::attributor {

class Attributer;

ChangeStatus AbstractElement::update(Attributor &solver) {
  ChangeStatus changeStatus = ChangeStatus::UNCHANGED;
  if (getState().isAtFixpoint())
    return changeStatus;

  changeStatus = updateImpl(solver);

  return changeStatus;
}

} // namespace rust_compiler::analysis::attributer
