#include "PatternDeclaration.h"

#include "ADT/CanonicalPath.h"
#include "AST/Patterns/IdentifierPattern.h"
#include "AST/Patterns/PatternWithoutRange.h"
#include "Resolver.h"

#include <memory>

using namespace rust_compiler::ast::patterns;

namespace rust_compiler::sema::resolver {

void PatternDeclaration::resolve() {
  switch (pat->getKind()) {
  case PatternNoTopAltKind::PatternWithoutRange: {
    resolvePatternWithoutRange(
        std::static_pointer_cast<PatternWithoutRange>(pat));
    break;
  }
  case PatternNoTopAltKind::RangePattern: {
    assert(false && "to be handled later");
    break;
  }
  }
}

void PatternDeclaration::resolvePatternWithoutRange(
    std::shared_ptr<ast::patterns::PatternWithoutRange> woRange) {
  switch (woRange->getWithoutRangeKind()) {
  case PatternWithoutRangeKind::LiteralPattern: {
    assert(false && "to be handled later");
  }
  case PatternWithoutRangeKind::IdentifierPattern: {
    resolveIdentifierPattern(
        std::static_pointer_cast<IdentifierPattern>(woRange));
    break;
  }
  case PatternWithoutRangeKind::WildcardPattern: {
    assert(false && "to be handled later");
  }
  case PatternWithoutRangeKind::RestPattern: {
    assert(false && "to be handled later");
  }
  case PatternWithoutRangeKind::ReferencePattern: {
    assert(false && "to be handled later");
  }
  case PatternWithoutRangeKind::StructPattern: {
    assert(false && "to be handled later");
  }
  case PatternWithoutRangeKind::TupleStructPattern: {
    assert(false && "to be handled later");
  }
  case PatternWithoutRangeKind::TuplePattern: {
    assert(false && "to be handled later");
  }
  case PatternWithoutRangeKind::GroupedPattern: {
    assert(false && "to be handled later");
  }
  case PatternWithoutRangeKind::SlicePattern: {
    assert(false && "to be handled later");
  }
  case PatternWithoutRangeKind::PathPattern: {
    assert(false && "to be handled later");
  }
  case PatternWithoutRangeKind::MacroInvocation: {
    assert(false && "to be handled later");
  }
  }
}

void PatternDeclaration::resolveIdentifierPattern(
    std::shared_ptr<ast::patterns::IdentifierPattern> id) {
  Mutability mut = id->hasMut() ? Mutability::Mutable : Mutability::Immutable;
  addNewBinding(id->getIdentifier(), id->getNodeId(),
                BindingTypeInfo(mut, id->hasRef(), id->getLocation()));
}

void PatternDeclaration::addNewBinding(std::string_view name, basic::NodeId id,
                                       BindingTypeInfo bind) {
  assert(bindings.size() > 0);

  bool identifierOrBound = false;
  bool identifierProductBound = false;

  for (auto binding : bindings) {
    if (binding.idents.find(std::string(name)) != binding.idents.end()) {
      identifierProductBound |= binding.ctx == PatternBoundCtx::Product;
      identifierOrBound |= binding.ctx == PatternBoundCtx::Or;
    }
  }

  if (identifierProductBound) {
    if (rib == RibKind::Parameter) {
      // report error
    } else {
      // report error
    }
  }

  if (!identifierOrBound) {
    bindings.back().idents.insert(std::string(name));
    resolver->getNameScope().insert(adt::CanonicalPath::newSegment(id, name),
                                    id, bind.getLocation(), rib);
  }

  bindingInfoMap.insert({std::string(name), bind});
}

} // namespace rust_compiler::sema::resolver
