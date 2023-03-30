#include "AttributeChecker.h"

#include "AST/InnerAttribute.h"

using namespace rust_compiler::ast;

namespace rust_compiler::sema::attribute_checker {

void AttributeChecker::checkCrate(std::shared_ptr<ast::Crate> crate) {
  assert(false);

  std::vector<InnerAttribute> inner = crate->getInnerAttributes();

  for (auto&inn: inner) {
    Attr& attr = inn.getAttr();
  }
}

} // namespace rust_compiler::sema::attribute_checker
