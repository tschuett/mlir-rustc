#pragma once

#include "AST/Crate.h"

#include <memory>

namespace rust_compiler::ast {
class Struct;
class Trait;
class TypeAlias;
class Enumeration;
class Implementation;
class InherentImpl;
class TraitImpl;
} // namespace rust_compiler::ast

namespace rust_compiler::sema::attribute_checker {

using namespace rust_compiler::ast;

/// https://doc.rust-lang.org/reference/attributes.html

class AttributeChecker {
public:
  void checkCrate(std::shared_ptr<ast::Crate> crate);

private:
  void checkItem(Item *item);
  void checkVisItem(VisItem *item);
  void checkFunction(Function *item);
  void checkStruct(Struct *item);
  void checkTrait(Trait *);
  void checkTypeAlias(TypeAlias *);
  void checkEnumeration(Enumeration *);
  void checkImplementation(Implementation *);
  void checkInherentImpl(InherentImpl *);
  void checkTraitImpl(TraitImpl *);
};

} // namespace rust_compiler::sema::attribute_checker
