#include "AttributeChecker.h"

namespace rust_compiler::sema::attribute_checker {

void AttributeChecker::checkFunction(Function *item) {

  //  for (auto &outer : item->getOuterAttributes()) {
  //  }

  // assert(false);
}

void AttributeChecker::checkStruct(Struct *item) {}

void AttributeChecker::checkTrait(Trait *) {}

} // namespace rust_compiler::sema::attribute_checker
