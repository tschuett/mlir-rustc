#include "AST/Implementation.h"
#include "AST/InherentImpl.h"
#include "AST/TraitImpl.h"
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

void AttributeChecker::checkImplementation(Implementation *impl) {
  switch (impl->getKind()) {
  case ImplementationKind::InherentImpl: {
    checkInherentImpl(static_cast<InherentImpl *>(impl));
    break;
  }
  case ImplementationKind::TraitImpl: {
    checkTraitImpl(static_cast<TraitImpl *>(impl));
    break;
  }
  }
}

void AttributeChecker::checkInherentImpl(InherentImpl *) {}
void AttributeChecker::checkTraitImpl(TraitImpl *) {}

} // namespace rust_compiler::sema::attribute_checker
