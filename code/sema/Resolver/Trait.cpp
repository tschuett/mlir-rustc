#include "ADT/CanonicalPath.h"
#include "AST/FunctionParam.h"
#include "AST/FunctionParamPattern.h"
#include "AST/FunctionParameters.h"
#include "AST/SelfParam.h"
#include "AST/TypedSelf.h"
#include "Basic/Ids.h"
#include "Lexer/Identifier.h"
#include "PatternDeclaration.h"
#include "Resolver.h"

#include <memory>

using namespace rust_compiler::adt;
using namespace rust_compiler::ast;

namespace rust_compiler::sema::resolver {

void Resolver::resolveMacroInvocationSemiInTrait(
    ast::MacroInvocationSemiItem *, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  assert(false);
}

void Resolver::resolveTypeAliasInTrait(
    ast::TypeAlias *, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  assert(false);
}

void Resolver::resolveConstantItemInTrait(
    ast::ConstantItem *, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  assert(false);
}

void Resolver::resolveFunctionInTrait(
    ast::Function *f, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  CanonicalPath decl = CanonicalPath::newSegment(f->getNodeId(), f->getName());
  CanonicalPath path = prefix.append(decl);
  CanonicalPath cpath = canonicalPrefix.append(decl);

  // FIXME handle identifier

  tyCtx->insertCanonicalPath(f->getNodeId(), cpath);

  basic::NodeId scopeNodeId = f->getNodeId();
  getNameScope().push(scopeNodeId);
  getTypeScope().push(scopeNodeId);
  getLabelScope().push(scopeNodeId);
  pushNewNameRib(getNameScope().peek());
  pushNewTypeRib(getTypeScope().peek());
  pushNewLabelRib(getLabelScope().peek());

  if (f->hasGenericParams())
    resolveGenericParams(f->getGenericParams(), prefix, canonicalPrefix);
  if (f->hasReturnType())
    resolveType(f->getReturnType(), prefix, canonicalPrefix);

  ast::FunctionParameters params = f->getParams();
  if (f->hasParams()) {
    if (params.hasSelfParam()) {
      // should first
      SelfParam self = params.getSelfParam();
      SelfParamKind kind = self.getKind();
      std::shared_ptr<ast::SelfParam> seFl = self.getSelf();
      switch (kind) {
      case SelfParamKind::ShorthandSelf: {
        // FIXME
        break;
      }
      case SelfParamKind::TypeSelf: {
        resolveType(std::static_pointer_cast<TypedSelf>(seFl)->getType(),
                    prefix, canonicalPrefix);
        break;
      }
      }
    }
  }
  // handle maybe Self

  std::vector<PatternBinding> bindings = {
      PatternBinding(PatternBoundCtx::Product, std::set<basic::NodeId>())};

  for (auto &param : params.getParams()) {
    switch (param.getKind()) {
    case FunctionParamKind::Pattern: {
      FunctionParamPattern pattern = param.getPattern();
      if (pattern.hasType())
        resolveType(pattern.getType(), prefix, canonicalPrefix);
      PatternDeclaration pat = {
          pattern.getPattern(), RibKind::Parameter, bindings, this, prefix,
          canonicalPrefix};
      pat.resolve();
      //
      //      resolvePatternDeclarationWithBindings(pattern.getPattern(),
      //                                            RibKind::Parameter,
      //                                            bindings, prefix,
      //                                            canonicalPrefix);
      break;
    }
    case FunctionParamKind::DotDotDot: {
      // should be last!
      assert(false);
      break;
    }
    case FunctionParamKind::Type: {
      resolveType(param.getType(), prefix, canonicalPrefix);
      break;
    }
    }
  }

  if (f->hasWhereClause())
    resolveWhereClause(f->getWhereClause(), prefix, canonicalPrefix);
  if (f->hasBody())
    resolveExpression(f->getBody(), path, cpath);

  getNameScope().pop();
  getTypeScope().pop();
  getLabelScope().pop();
}

} // namespace rust_compiler::sema::resolver
