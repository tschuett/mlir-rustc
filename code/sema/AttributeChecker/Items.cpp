#include "AttributeChecker.h"

namespace rust_compiler::sema::attribute_checker {

void AttributeChecker::checkFunction(Function *item) {

  //  for (auto &outer : item->getOuterAttributes()) {
  //  }

  // assert(false);
}

void AttributeChecker::checkStruct(Struct *item) {}

void AttributeChecker::checkTrait(Trait *) {}

void AttributeChecker::checkTypeAlias(TypeAlias *) {}

void AttributeChecker::checkEnumeration(Enumeration *) {}

} // namespace rust_compiler::sema::attribute_checker
