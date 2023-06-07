#include "PatternDeclaration.h"

#include "ADT/CanonicalPath.h"
#include "AST/Patterns/IdentifierPattern.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Patterns/PatternWithoutRange.h"
#include "AST/Patterns/TupleStructItems.h"
#include "AST/Patterns/TupleStructPattern.h"
#include "Resolver.h"

#include <llvm/Support/FormatVariadic.h>
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
    break;
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
    resolveTupleStructPattern(
        std::static_pointer_cast<TupleStructPattern>(woRange));
    break;
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

void PatternDeclaration::resolveTupleStructPattern(
    std::shared_ptr<ast::patterns::TupleStructPattern> path) {
  resolver->resolveExpression(path->getPath(), prefix, canonicalPrefix);

  if (path->hasItems()) {
    TupleStructItems items = path->getItems();
    for (auto &pat : items.getPatterns()) {
      for (auto &p : pat->getPatterns()) {
        switch (p->getKind()) {
        case PatternNoTopAltKind::RangePattern: {
          assert(false);
        }
        case PatternNoTopAltKind::PatternWithoutRange: {
          std::shared_ptr<ast::patterns::PatternWithoutRange> wo =
              std::static_pointer_cast<ast::patterns::PatternWithoutRange>(p);
          resolvePatternWithoutRange(wo);
          break;
        }
        }
      }
    }
  }
}

void PatternDeclaration::addNewBinding(const lexer::Identifier &name,
                                       basic::NodeId id, BindingTypeInfo bind) {
  assert(bindings.size() > 0);

  bool identifierOrBound = false;
  bool identifierProductBound = false;

  for (auto binding : bindings) {
    if (binding.contains(id)) {
      identifierProductBound |= binding.getCtx() == PatternBoundCtx::Product;
      identifierOrBound |= binding.getCtx() == PatternBoundCtx::Or;
    }
  }

  if (identifierProductBound) {
    if (rib == RibKind::Parameter) {
      // report error
      llvm::errs() << llvm::formatv("identifier {0} is bound more than once in "
                                    "the same parameter list",
                                    name.toString())
                          .str()
                   << "\n";
    } else {
      // report error
      llvm::errs() << llvm::formatv("identifier {0} is bound more than once in "
                                    "the same pattern",
                                    name.toString())
                          .str()
                   << "\n";
    }
  }

  if (!identifierOrBound) {
    bindings.back().insert(id);
    resolver->getNameScope().insert(adt::CanonicalPath::newSegment(id, name),
                                    id, bind.getLocation(), rib);
  }

  bindingInfoMap.insert({name, bind});
}

} // namespace rust_compiler::sema::resolver
