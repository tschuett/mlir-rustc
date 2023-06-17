#include "ADT/CanonicalPath.h"
#include "AST/Expression.h"
#include "AST/PathExpression.h"
#include "AST/PathInExpression.h"
#include "AST/Patterns/IdentifierPattern.h"
#include "AST/Patterns/Pattern.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Patterns/PatternWithoutRange.h"
#include "AST/Patterns/StructPatternElements.h"
#include "AST/Patterns/TupleStructItems.h"
#include "AST/Patterns/TupleStructPattern.h"
#include "Basic/Ids.h"
#include "PatternDeclaration.h"
#include "Resolver.h"
#include "llvm/Support/ErrorHandling.h"

#include <memory>

using namespace rust_compiler::ast::patterns;
using namespace rust_compiler::adt;
using namespace rust_compiler::ast;

namespace rust_compiler::sema::resolver {

//void Resolver::resolvePatternDeclaration(
//    std::shared_ptr<ast::patterns::PatternNoTopAlt> pattern, RibKind kind,
//    const adt::CanonicalPath &prefix,
//    const adt::CanonicalPath &canonicalPrefix) {
//  std::vector<PatternBinding> bindings = {
//      PatternBinding(PatternBoundCtx::Product, std::set<basic::NodeId>())};
//
//  resolvePatternDeclarationWithBindings(pattern, kind, bindings, prefix,
//                                        canonicalPrefix);
//}

//void Resolver::resolvePatternDeclaration(
//    std::shared_ptr<ast::patterns::Pattern> pattern, RibKind kind,
//    const adt::CanonicalPath &prefix,
//    const adt::CanonicalPath &canonicalPrefix) {
//  std::vector<PatternBinding> bindings = {
//      PatternBinding(PatternBoundCtx::Product, std::set<basic::NodeId>())};
//
//  for (auto &p : pattern->getPatterns())
//    resolvePatternDeclarationWithBindings(p, kind, bindings, prefix,
//                                          canonicalPrefix);
//}
//
//void Resolver::resolvePatternDeclarationWithBindings(
//    std::shared_ptr<ast::patterns::PatternNoTopAlt> noTopAlt, RibKind ribKind,
//    std::vector<PatternBinding> &bindings, const adt::CanonicalPath &prefix,
//    const adt::CanonicalPath &canonicalPrefix) {
//  switch (noTopAlt->getKind()) {
//  case PatternNoTopAltKind::PatternWithoutRange: {
//    resolvePatternDeclarationWithoutRange(
//        std::static_pointer_cast<PatternWithoutRange>(noTopAlt), ribKind,
//        bindings, prefix, canonicalPrefix);
//    break;
//  }
//  case PatternNoTopAltKind::RangePattern: {
//    assert(false && "to be handled later");
//  }
//  }
//}
//
//void Resolver::resolvePatternDeclarationWithoutRange(
//    std::shared_ptr<ast::patterns::PatternWithoutRange> pat, RibKind rib,
//    std::vector<PatternBinding> &bindings, const adt::CanonicalPath &prefix,
//    const adt::CanonicalPath &canonicalPrefix) {
//  switch (pat->getWithoutRangeKind()) {
//  case PatternWithoutRangeKind::LiteralPattern: {
//    assert(false && "to be handled later");
//  }
//  case PatternWithoutRangeKind::IdentifierPattern: {
//    std::shared_ptr<ast::patterns::IdentifierPattern> id =
//        std::static_pointer_cast<IdentifierPattern>(pat);
//    getNameScope().insert(
//        CanonicalPath::newSegment(id->getNodeId(), id->getIdentifier()),
//        id->getNodeId(), id->getLocation(), rib);
//    break;
//  }
//  case PatternWithoutRangeKind::WildcardPattern: {
//    assert(false && "to be handled later");
//  }
//  case PatternWithoutRangeKind::RestPattern: {
//    assert(false && "to be handled later");
//  }
//  case PatternWithoutRangeKind::ReferencePattern: {
//    assert(false && "to be handled later");
//  }
//  case PatternWithoutRangeKind::StructPattern: {
//    resolveStructPatternDeclaration(
//        std::static_pointer_cast<StructPattern>(pat), rib, bindings, prefix,
//        canonicalPrefix);
//    break;
//  }
//  case PatternWithoutRangeKind::TupleStructPattern: {
//    resolveTupleStructPatternDeclaration(
//        std::static_pointer_cast<TupleStructPattern>(pat), rib, bindings,
//        prefix, canonicalPrefix);
//    break;
//  }
//  case PatternWithoutRangeKind::TuplePattern: {
//    assert(false && "to be handled later");
//  }
//  case PatternWithoutRangeKind::GroupedPattern: {
//    assert(false && "to be handled later");
//  }
//  case PatternWithoutRangeKind::SlicePattern: {
//    assert(false && "to be handled later");
//  }
//  case PatternWithoutRangeKind::PathPattern: {
//    resolvePathPatternDeclaration(std::static_pointer_cast<PathPattern>(pat),
//                                  rib, bindings, prefix, canonicalPrefix);
//    break;
//  }
//  case PatternWithoutRangeKind::MacroInvocation: {
//    assert(false && "to be handled later");
//  }
//  }
//}
//
//void Resolver::resolvePathPatternDeclaration(
//    std::shared_ptr<ast::patterns::PathPattern> pat, RibKind rib,
//    std::vector<PatternBinding> &bindings, const adt::CanonicalPath &prefix,
//    const adt::CanonicalPath &canonicalPrefix) {
//  std::shared_ptr<ast::PathExpression> path =
//      std::static_pointer_cast<PathExpression>(pat->getPath());
//
//  switch (path->getPathExpressionKind()) {
//  case PathExpressionKind::PathInExpression: {
//    resolvePathInExpression(
//        std::static_pointer_cast<ast::PathInExpression>(path), prefix,
//        canonicalPrefix);
//    break;
//  }
//  case PathExpressionKind::QualifiedPathInExpression: {
//    assert(false && "to be handled later");
//  }
//  }
//
//  //  if (path->getExpressionKind() ==
//  //      ast::ExpressionKind::ExpressionWithoutBlock) {
//  //    std::shared_ptr<ast::ExpressionWithoutBlock> pathWoBlock =
//  //        std::static_pointer_cast<ast::ExpressionWithoutBlock>(path);
//  //    if (pathWoBlock->getWithoutBlockKind() ==
//  //        ast::ExpressionWithoutBlockKind::PathExpression) {
//  //      std::shared_ptr<ast::PathExpression> pathExpr =
//  //          std::static_pointer_cast<ast::PathExpression>(pathWoBlock);
//  //      if (pathExpr->getPathExpressionKind() ==
//  //          ast::PathExpressionKind::PathInExpression) {
//  //        resolvePathExpression(
//  //            std::static_pointer_cast<ast::PathInExpression>(pathExpr));
//  //      }
//  //    }
//  //  }
//}
//
//void Resolver::resolveTupleStructPatternDeclaration(
//    std::shared_ptr<ast::patterns::TupleStructPattern> tuple, RibKind rib,
//    std::vector<PatternBinding> &bindings, const adt::CanonicalPath &prefix,
//    const adt::CanonicalPath &canonicalPrefix) {
//  resolveExpression(tuple->getPath(), prefix, canonicalPrefix);
//
//  if (tuple->hasItems()) {
//    TupleStructItems items = tuple->getItems();
//    for (const auto &pat : items.getPatterns()) {
//      for (const auto &pat2 : pat->getPatterns()) {
//        switch (pat2->getKind()) {
//        case PatternNoTopAltKind::RangePattern: {
//          llvm_unreachable("there are no range patterns");
//        }
//        case PatternNoTopAltKind::PatternWithoutRange: {
//          resolvePatternDeclarationWithBindings(
//              std::static_pointer_cast<patterns::PatternNoTopAlt>(pat2), rib,
//              bindings, prefix, canonicalPrefix);
//          break;
//        }
//        }
//      }
//    }
//  }
//}
//
//void Resolver::resolveStructPatternDeclaration(
//    std::shared_ptr<ast::patterns::StructPattern> stru, RibKind rib,
//    std::vector<PatternBinding> &bindings, const adt::CanonicalPath &prefix,
//    const adt::CanonicalPath &canonicalPrefix) {
//  resolveExpression(stru->getPath(), prefix, canonicalPrefix);
//
//  if (stru->hasElement()) {
//    StructPatternElements el = stru->getElements();
//    if (el.hasElements()) {
//      StructPatternFields fie = el.getFields();
//      for (const StructPatternField &f : fie.getFields()) {
//        switch(f.getKind()) {
//        case StructPatternFieldKind::TupleIndex: {
//        }
//        case StructPatternFieldKind::Identifier: {
//        }
//        case StructPatternFieldKind::RefMut: {
//        }
//        }
//        for (const& noTop: f.getPattern().getPatterns()) {
//          
//        }
//      }
//    }
//  }
//}

} // namespace rust_compiler::sema::resolver
