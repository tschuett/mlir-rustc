#pragma once

#include "ADT/CanonicalPath.h"
#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/ArrayExpression.h"
#include "AST/AssociatedItem.h"
#include "AST/BorrowExpression.h"
#include "AST/CallExpression.h"
#include "AST/ClosureExpression.h"
#include "AST/ComparisonExpression.h"
#include "AST/ConstantItem.h"
#include "AST/Crate.h"
#include "AST/DereferenceExpression.h"
#include "AST/Expression.h"
#include "AST/ExpressionStatement.h"
#include "AST/FieldExpression.h"
#include "AST/Function.h"
#include "AST/IfExpression.h"
#include "AST/IfLetExpression.h"
#include "AST/Implementation.h"
#include "AST/IndexEpression.h"
#include "AST/InfiniteLoopExpression.h"
#include "AST/InherentImpl.h"
#include "AST/ItemDeclaration.h"
#include "AST/IteratorLoopExpression.h"
#include "AST/LetStatement.h"
#include "AST/LoopExpression.h"
#include "AST/MacroInvocationSemiItem.h"
#include "AST/MacroItem.h"
#include "AST/MatchExpression.h"
#include "AST/MethodCallExpression.h"
#include "AST/OperatorExpression.h"
#include "AST/PathExpression.h"
#include "AST/PathIdentSegment.h"
#include "AST/PathInExpression.h"
#include "AST/Patterns/PathPattern.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Patterns/PatternWithoutRange.h"
#include "AST/Patterns/TupleStructPattern.h"
#include "AST/QualifiedPathInExpression.h"
#include "AST/RangeExpression.h"
#include "AST/ReturnExpression.h"
#include "AST/SimplePath.h"
#include "AST/Statement.h"
#include "AST/StaticItem.h"
#include "AST/Struct.h"
#include "AST/StructExprFields.h"
#include "AST/StructExpression.h"
#include "AST/StructStruct.h"
#include "AST/Trait.h"
#include "AST/TraitImpl.h"
#include "AST/TupleExpression.h"
#include "AST/TupleIndexingExpression.h"
#include "AST/TupleStruct.h"
#include "AST/TypeAlias.h"
#include "AST/TypeCastExpression.h"
#include "AST/Types/ArrayType.h"
#include "AST/Types/ImplTraitType.h"
#include "AST/Types/ImplTraitTypeOneBound.h"
#include "AST/Types/RawPointerType.h"
#include "AST/Types/ReferenceType.h"
#include "AST/Types/SliceType.h"
#include "AST/Types/TraitObjectType.h"
#include "AST/Types/TupleType.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypeNoBounds.h"
#include "AST/Types/TypeParamBound.h"
#include "AST/Types/TypePathFn.h"
#include "AST/Types/TypePathSegment.h"
#include "AST/Union.h"
#include "AST/UnsafeBlockExpression.h"
#include "AST/UseDeclaration.h"
#include "AST/VisItem.h"
#include "AST/Visiblity.h"
#include "Basic/Ids.h"
#include "Location.h"
#include "TyCtx/TyCtx.h"

#include <cassert>
#include <map>
#include <optional>
#include <set>
#include <stack>
#include <string_view>
#include <vector>

namespace rust_compiler::ast {
class MethodCallExpression;
namespace types {
class TraitObjectTypeOneBound;
class TraitBound;
} // namespace types
} // namespace rust_compiler::ast

namespace rust_compiler::sema::resolver {

// Specifies whether the set of already bound patterns are related by 'Or' or
// 'Product'. Used to check for multiple bindings to the same identifier.
enum class PatternBoundCtx {
  // A product pattern context (e.g. struct and tuple patterns)
  Product,
  // An or-pattern context (e.g. p_0 | p_1 | ...)
  Or,
};

class PatternBinding {
  PatternBoundCtx ctx;
  std::set<basic::NodeId> idents;

public:
  PatternBinding(PatternBoundCtx ctx, std::set<basic::NodeId> idents)
      : ctx(ctx), idents(idents) {}

  bool contains(NodeId id) const { return idents.find(id) != idents.end(); }

  PatternBoundCtx getCtx() const { return ctx; }

  void insert(NodeId id) { idents.insert(id); }
};

/// https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/late/index.html
/// https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/index.html
///  https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/struct.Resolver.html

/// https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/late/struct.Rib.html

///  https://github.com/Rust-GCC/gccrs/pull/2344
enum class RibKind {
  Dummy,
  Function,
  Label,
  Parameter,
  Unkown,
  Type,
  Variable,
  Trait,
  Module,
  Constant
};

class Rib {
public:
  // https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/late/enum.RibKind.html

  Rib(basic::CrateNum crateNum, basic::NodeId nodeId)
      : crateNum(crateNum), nodeId(nodeId) {}

  basic::NodeId getNodeId() const { return nodeId; }
  basic::CrateNum getCrateNum() const { return crateNum; }

  void insertName(const adt::CanonicalPath &path, basic::NodeId id,
                  Location loc, bool shadow, RibKind kind);
  std::optional<basic::NodeId> lookupName(const adt::CanonicalPath &ident);
  void appendReferenceForDef(basic::NodeId ref, basic::NodeId def);
  bool wasDeclDeclaredHere(basic::NodeId def) const;
  std::optional<RibKind> lookupDeclType(basic::NodeId id);

  void clearName(const adt::CanonicalPath &path, basic::NodeId id);

  void print() const;

private:
  basic::CrateNum crateNum;
  basic::NodeId nodeId;
  std::map<adt::CanonicalPath, basic::NodeId> pathMappings;
  std::map<basic::NodeId, RibKind> declTypeMappings;
  std::map<basic::NodeId, std::set<basic::NodeId>> references;
  std::map<basic::NodeId, Location> declsWithinRib;
  std::map<basic::NodeId, adt::CanonicalPath> reversePathMappings;
};

class Scope {
public:
  Scope(basic::CrateNum crateNum) : crateNum(crateNum) {}

  Rib *peek();
  void push(basic::NodeId id);
  Rib *pop();

  basic::CrateNum getCrateNum() const { return crateNum; }

  void insert(const adt::CanonicalPath &, basic::NodeId, Location,
              RibKind kind = RibKind::Unkown);
  void appendReferenceForDef(basic::NodeId ref, basic::NodeId def);

  bool wasDeclDeclaredInCurrentScope(basic::NodeId def) const;
  std::optional<basic::NodeId> lookup(const adt::CanonicalPath &p);

  std::optional<RibKind> lookupDeclType(basic::NodeId id);
  std::optional<Rib *> lookupRibForDecl(basic::NodeId id);

  const std::vector<Rib *> &getContext() const { return stack; };

  void print() const;

private:
  basic::CrateNum crateNum;
  // basic::NodeId nodeId;
  std::vector<Rib *> stack;
};

class Segment {
  std::string name;
};

class Import {
public:
  enum class ImportKind { Single, Glob, ExternCrate, MacroUse, MacroExport };

  ImportKind getKind() const { return kind; }

  basic::NodeId getNodeId() const { return nodeId; }

private:
  ImportKind kind;
  basic::NodeId nodeId;
  llvm::SmallVector<Segment> modulePath;
};

/// https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/late/struct.SelfVisitor.html
class Resolver {
public:
  Resolver() noexcept;

  ~Resolver() = default;

  void resolveCrate(std::shared_ptr<ast::Crate>);

  std::optional<basic::NodeId> lookupResolvedName(basic::NodeId nodeId);
  std::optional<basic::NodeId> lookupResolvedType(basic::NodeId nodeId);

  Scope &getNameScope() { return nameScope; }
  Scope &getTypeScope() { return typeScope; }

  void resolveExpression(std::shared_ptr<ast::Expression>,
                         const adt::CanonicalPath &prefix,
                         const adt::CanonicalPath &canonicalPrefix);

  void insertResolvedType(basic::NodeId refId, basic::NodeId defId);
  void insertResolvedMisc(NodeId refId, NodeId defId);

private:
  // items no recurse
  void resolveItemNoRecurse(std::shared_ptr<ast::Item>,
                            const adt::CanonicalPath &prefix,
                            const adt::CanonicalPath &canonicalPrefix);
  void resolveVisItemNoRecurse(std::shared_ptr<ast::VisItem>,
                               const adt::CanonicalPath &prefix,
                               const adt::CanonicalPath &canonicalPrefix);
  void resolveMacroItemNoRecurse(std::shared_ptr<ast::MacroItem>,
                                 const adt::CanonicalPath &prefix,
                                 const adt::CanonicalPath &canonicalPrefix);
  void resolveFunctionNoRecurse(std::shared_ptr<ast::Function>,
                                const adt::CanonicalPath &prefix,
                                const adt::CanonicalPath &canonicalPrefix);
  void resolveStructNoRecurse(ast::Struct *, const adt::CanonicalPath &prefix,
                              const adt::CanonicalPath &canonicalPrefix);
  void resolveStructStructNoRecurse(ast::StructStruct *,
                                    const adt::CanonicalPath &prefix,
                                    const adt::CanonicalPath &canonicalPrefix);
  void resolveTupleStructNoRecurse(ast::TupleStruct *,
                                   const adt::CanonicalPath &prefix,
                                   const adt::CanonicalPath &canonicalPrefix);
  void resolveTypeAliasNoRecurse(ast::TypeAlias *,
                                 const adt::CanonicalPath &prefix,
                                 const adt::CanonicalPath &canonicalPrefix);
  void resolveTraitNoRecurse(std::shared_ptr<ast::Trait> trait,
                             const adt::CanonicalPath &prefix,
                             const adt::CanonicalPath &canonicalPrefix);
  void resolveModuleNoRecurse(std::shared_ptr<ast::Module> module,
                              const adt::CanonicalPath &prefix,
                              const adt::CanonicalPath &canonicalPrefix);
  void resolveEnumerationNoRecurse(ast::Enumeration *,
                                   const adt::CanonicalPath &prefix,
                                   const adt::CanonicalPath &canonicalPrefix);
  void resolveImplementationNoRecurse(
      std::shared_ptr<ast::Implementation> implementation,
      const adt::CanonicalPath &prefix,
      const adt::CanonicalPath &canonicalPrefix);
  void resolveInherentImplNoRecurse(
      std::shared_ptr<ast::InherentImpl> implementation,
      const adt::CanonicalPath &prefix,
      const adt::CanonicalPath &canonicalPrefix);
  void resolveTraitImplNoRecurse(std::shared_ptr<ast::TraitImpl> implementation,
                                 const adt::CanonicalPath &prefix,
                                 const adt::CanonicalPath &canonicalPrefix);

  // items
  void resolveItem(std::shared_ptr<ast::Item>, const adt::CanonicalPath &prefix,
                   const adt::CanonicalPath &canonicalPrefix);
  void resolveVisItem(std::shared_ptr<ast::VisItem>,
                      const adt::CanonicalPath &prefix,
                      const adt::CanonicalPath &canonicalPrefix);
  void resolveMacroItem(std::shared_ptr<ast::MacroItem>,
                        const adt::CanonicalPath &prefix,
                        const adt::CanonicalPath &canonicalPrefix);
  void resolveStaticItem(std::shared_ptr<ast::StaticItem>,
                         const adt::CanonicalPath &prefix,
                         const adt::CanonicalPath &canonicalPrefix);
  void resolveConstantItem(std::shared_ptr<ast::ConstantItem>,
                           const adt::CanonicalPath &prefix,
                           const adt::CanonicalPath &canonicalPrefix);
  void resolveImplementation(std::shared_ptr<ast::Implementation>,
                             const adt::CanonicalPath &prefix,
                             const adt::CanonicalPath &canonicalPrefix);
  void resolveUseDeclaration(std::shared_ptr<ast::UseDeclaration>,
                             const adt::CanonicalPath &prefix,
                             const adt::CanonicalPath &canonicalPrefix);
  void resolveInherentImpl(std::shared_ptr<ast::InherentImpl>,
                           const adt::CanonicalPath &prefix,
                           const adt::CanonicalPath &canonicalPrefix);
  void resolveTraitImpl(std::shared_ptr<ast::TraitImpl>,
                        const adt::CanonicalPath &prefix,
                        const adt::CanonicalPath &canonicalPrefix);
  void resolveFunction(ast::Function *, const adt::CanonicalPath &prefix,
                       const adt::CanonicalPath &canonicalPrefix);
  void resolveModule(std::shared_ptr<ast::Module>,
                     const adt::CanonicalPath &prefix,
                     const adt::CanonicalPath &canonicalPrefix);
  void resolveStructItem(std::shared_ptr<ast::Struct>,
                         const adt::CanonicalPath &prefix,
                         const adt::CanonicalPath &canonicalPrefix);
  void resolveTypeAlias(ast::TypeAlias *, const adt::CanonicalPath &prefix,
                        const adt::CanonicalPath &canonicalPrefix);
  void
  resolveStructStructItem(std::shared_ptr<rust_compiler::ast::StructStruct>,
                          const adt::CanonicalPath &prefix,
                          const adt::CanonicalPath &canonicalPrefix);
  void resolveTupleStructItem(std::shared_ptr<ast::TupleStruct>,
                              const adt::CanonicalPath &prefix,
                              const adt::CanonicalPath &canonicalPrefix);
  void resolveEnumerationItem(std::shared_ptr<ast::Enumeration>,
                              const adt::CanonicalPath &prefix,
                              const adt::CanonicalPath &canonicalPrefix);
  void resolveEnumItem(std::shared_ptr<ast::EnumItem>,
                       const adt::CanonicalPath &prefix,
                       const adt::CanonicalPath &canonicalPrefix);
  void resolveTraitItem(std::shared_ptr<ast::Trait>,
                        const adt::CanonicalPath &prefix,
                        const adt::CanonicalPath &canonicalPrefix);
  void resolveAssociatedItem(ast::AssociatedItem *,
                             const adt::CanonicalPath &prefix,
                             const adt::CanonicalPath &canonicalPrefix,
                             NodeId implementationId);
  void resolveUnionItem(std::shared_ptr<ast::Union>,
                        const adt::CanonicalPath &prefix,
                        const adt::CanonicalPath &canonicalPrefix);
  void resolveItemDeclaration(std::shared_ptr<ast::ItemDeclaration>,
                              const adt::CanonicalPath &prefix,
                              const adt::CanonicalPath &canonicalPrefix);
  void resolveConstantItemNoRecurse(ast::ConstantItem *,
                                    const adt::CanonicalPath &prefix,
                                    const adt::CanonicalPath &canonicalPrefix);
  void resolveExpressionNoRecurse(ast::Expression *);
  void resolveExpressionWithBlockNoRecurse(ast::ExpressionWithBlock *);
  void resolveBlockExpressionNoRecurse(ast::BlockExpression *);

  // expressions
  void resolveExpressionWithBlock(std::shared_ptr<ast::ExpressionWithBlock>,
                                  const adt::CanonicalPath &prefix,
                                  const adt::CanonicalPath &canonicalPrefix);
  void
  resolveExpressionWithoutBlock(std::shared_ptr<ast::ExpressionWithoutBlock>,
                                const adt::CanonicalPath &prefix,
                                const adt::CanonicalPath &canonicalPrefix);
  void resolveReturnExpression(std::shared_ptr<ast::ReturnExpression>,
                               const adt::CanonicalPath &prefix,
                               const adt::CanonicalPath &canonicalPrefix);
  void resolveRangeExpression(std::shared_ptr<ast::RangeExpression>,
                              const adt::CanonicalPath &prefix,
                              const adt::CanonicalPath &canonicalPrefix);
  void resolveOperatorExpression(std::shared_ptr<ast::OperatorExpression>,
                                 const adt::CanonicalPath &prefix,
                                 const adt::CanonicalPath &canonicalPrefix);
  void resolveLoopExpression(std::shared_ptr<ast::LoopExpression>,
                             const adt::CanonicalPath &prefix,
                             const adt::CanonicalPath &canonicalPrefix);
  void resolveIfExpression(std::shared_ptr<ast::IfExpression>,
                           const adt::CanonicalPath &prefix,
                           const adt::CanonicalPath &canonicalPrefix);
  void resolveIfLetExpression(ast::IfLetExpression *,
                              const adt::CanonicalPath &prefix,
                              const adt::CanonicalPath &canonicalPrefix);
  void resolvePathExpression(std::shared_ptr<ast::PathExpression>,
                             const adt::CanonicalPath &prefix,
                             const adt::CanonicalPath &canonicalPrefix);
  void resolveBlockExpression(std::shared_ptr<ast::BlockExpression>,
                              const adt::CanonicalPath &prefix,
                              const adt::CanonicalPath &canonicalPrefix);
  void resolveUnsafeBlockExpression(std::shared_ptr<ast::UnsafeBlockExpression>,
                                    const adt::CanonicalPath &prefix,
                                    const adt::CanonicalPath &canonicalPrefix);
  void resolveMatchExpression(std::shared_ptr<ast::MatchExpression>,
                              const adt::CanonicalPath &prefix,
                              const adt::CanonicalPath &canonicalPrefix);
  void resolveClosureExpression(std::shared_ptr<ast::ClosureExpression>,
                                const adt::CanonicalPath &prefix,
                                const adt::CanonicalPath &canonicalPrefix);
  void
  resolveInfiniteLoopExpression(std::shared_ptr<ast::InfiniteLoopExpression>,
                                const adt::CanonicalPath &prefix,
                                const adt::CanonicalPath &canonicalPrefix);
  void
  resolveIteratorLoopExpression(std::shared_ptr<ast::IteratorLoopExpression>,
                                const adt::CanonicalPath &prefix,
                                const adt::CanonicalPath &canonicalPrefix);
  void resolveQualifiedPathInExpression(
      std::shared_ptr<ast::QualifiedPathInExpression>,
      const adt::CanonicalPath &prefix,
      const adt::CanonicalPath &canonicalPrefix);
  void resolveArithmeticOrLogicalExpression(
      std::shared_ptr<ast::ArithmeticOrLogicalExpression>,
      const adt::CanonicalPath &prefix,
      const adt::CanonicalPath &canonicalPrefix);
  void resolveComparisonExpression(std::shared_ptr<ast::ComparisonExpression>,
                                   const adt::CanonicalPath &prefix,
                                   const adt::CanonicalPath &canonicalPrefix);
  void resolveTypeCastExpression(std::shared_ptr<ast::TypeCastExpression>,
                                 const adt::CanonicalPath &prefix,
                                 const adt::CanonicalPath &canonicalPrefix);

  void resolveDereferenceExpression(std::shared_ptr<ast::DereferenceExpression>,
                                    const adt::CanonicalPath &prefix,
                                    const adt::CanonicalPath &canonicalPrefix);
  void resolveBorrowExpression(std::shared_ptr<ast::BorrowExpression>,
                               const adt::CanonicalPath &prefix,
                               const adt::CanonicalPath &canonicalPrefix);
  void resolveArrayExpression(std::shared_ptr<ast::ArrayExpression>,
                              const adt::CanonicalPath &prefix,
                              const adt::CanonicalPath &canonicalPrefix);
  void resolveIndexExpression(std::shared_ptr<ast::IndexExpression>,
                              const adt::CanonicalPath &prefix,
                              const adt::CanonicalPath &canonicalPrefix);
  void resolveClosureParameter(ast::ClosureParam &param,
                               std::vector<PatternBinding> &bindings,
                               const adt::CanonicalPath &prefix,
                               const adt::CanonicalPath &canonicalPrefix);
  void resolveCallExpression(std::shared_ptr<ast::CallExpression>,
                             const adt::CanonicalPath &prefix,
                             const adt::CanonicalPath &canonicalPrefix);
  void resolveMethodCallExpression(ast::MethodCallExpression *,
                                   const adt::CanonicalPath &prefix,
                                   const adt::CanonicalPath &canonicalPrefix);
  void resolveTupleExpression(ast::TupleExpression *,
                              const adt::CanonicalPath &prefix,
                              const adt::CanonicalPath &canonicalPrefix);
  void resolveFieldExpression(ast::FieldExpression *,
                              const adt::CanonicalPath &prefix,
                              const adt::CanonicalPath &canonicalPrefix);
  void
  resolveTupleIndexingExpression(ast::TupleIndexingExpression *,
                                 const adt::CanonicalPath &prefix,
                                 const adt::CanonicalPath &canonicalPrefix);

  // types
  std::optional<basic::NodeId>
  resolveType(std::shared_ptr<ast::types::TypeExpression>,
              const adt::CanonicalPath &prefix,
              const adt::CanonicalPath &canonicalPrefix);
  std::optional<basic::NodeId>
  resolveTypeNoBounds(std::shared_ptr<ast::types::TypeNoBounds>,
                      const adt::CanonicalPath &prefix,
                      const adt::CanonicalPath &canonicalPrefix);
  std::optional<basic::NodeId>
  resolveRelativeTypePath(std::shared_ptr<ast::types::TypePath>,
                          const adt::CanonicalPath &prefix,
                          const adt::CanonicalPath &canonicalPrefix);
  void resolveTypePathFunction(const ast::types::TypePathFn &,
                               const adt::CanonicalPath &prefix,
                               const adt::CanonicalPath &canonicalPrefix);
  std::optional<basic::NodeId>
  resolveArrayType(std::shared_ptr<ast::types::ArrayType>,
                   const adt::CanonicalPath &prefix,
                   const adt::CanonicalPath &canonicalPrefix);
  std::optional<basic::NodeId>
  resolveTupleType(ast::types::TupleType *, const adt::CanonicalPath &prefix,
                   const adt::CanonicalPath &canonicalPrefix);
  std::optional<basic::NodeId>
  resolveReferenceType(std::shared_ptr<ast::types::ReferenceType>,
                       const adt::CanonicalPath &prefix,
                       const adt::CanonicalPath &canonicalPrefix);

  std::optional<basic::NodeId> resolveImplTraitTypeOneBound(
      std::shared_ptr<ast::types::ImplTraitTypeOneBound>,
      const adt::CanonicalPath &prefix,
      const adt::CanonicalPath &canonicalPrefix);
  std::optional<basic::NodeId> resolveTraitObjectTypeOneBound(
      std::shared_ptr<ast::types::TraitObjectTypeOneBound>,
      const adt::CanonicalPath &prefix,
      const adt::CanonicalPath &canonicalPrefix);
  std::optional<basic::NodeId>
  resolveTypeParamBound(std::shared_ptr<ast::types::TypeParamBound>,
                        const adt::CanonicalPath &prefix,
                        const adt::CanonicalPath &canonicalPrefix);
  std::optional<basic::NodeId>
  resolveRawPointerType(std::shared_ptr<ast::types::RawPointerType>,
                        const adt::CanonicalPath &prefix,
                        const adt::CanonicalPath &canonicalPrefix);
  std::optional<basic::NodeId>
  resolveSliceType(std::shared_ptr<ast::types::SliceType>,
                   const adt::CanonicalPath &prefix,
                   const adt::CanonicalPath &canonicalPrefix);

  std::optional<adt::CanonicalPath>
  resolveTypeToCanonicalPath(ast::types::TypeExpression *,
                             const adt::CanonicalPath &prefix,
                             const adt::CanonicalPath &canonicalPrefix);
  bool
  resolveTypeNoBoundsToCanonicalPath(ast::types::TypeNoBounds *,
                                     adt::CanonicalPath &result,
                                     const adt::CanonicalPath &prefix,
                                     const adt::CanonicalPath &canonicalPrefix);
  bool
  resolveTypePathToCanonicalPath(ast::types::TypePath *,
                                 adt::CanonicalPath &result,
                                 const adt::CanonicalPath &prefix,
                                 const adt::CanonicalPath &canonicalPrefix);

  std::string resolveTypeToString(ast::types::TypeExpression *);
  std::string resolveTypeNoBoundsToString(ast::types::TypeNoBounds *);
  std::string resolveImplTraitTypeToString(ast::types::ImplTraitType *);
  std::string resolveTraitObjectTypeToString(ast::types::TraitObjectType *);
  std::string resolveTypePathToString(ast::types::TypePath *);
  std::string
  resolveTypePathSegmentToString(const ast::types::TypePathSegment &);
  std::string resolvePathIdentSegmentToString(const ast::PathIdentSegment &);
  // checks
  void resolveVisibility(std::optional<ast::Visibility>);

  // generics
  void resolveWhereClause(const ast::WhereClause &,
                          const adt::CanonicalPath &prefix,
                          const adt::CanonicalPath &canonicalPrefix);
  void resolveGenericParams(const ast::GenericParams &,
                            const adt::CanonicalPath &prefix,
                            const adt::CanonicalPath &canonicalPrefix);
  void resolveGenericArgs(const ast::GenericArgs &,
                          const adt::CanonicalPath &prefix,
                          const adt::CanonicalPath &canonicalPrefix);

  // patterns
  //  void resolvePatternDeclarationWithBindings(
  //      std::shared_ptr<ast::patterns::PatternNoTopAlt>, RibKind,
  //      std::vector<PatternBinding> &bindings, const adt::CanonicalPath
  //      &prefix, const adt::CanonicalPath &canonicalPrefix);
  //  void
  //  resolvePatternDeclaration(std::shared_ptr<ast::patterns::PatternNoTopAlt>,
  //                            RibKind, const adt::CanonicalPath &prefix,
  //                            const adt::CanonicalPath &canonicalPrefix);
  //  void resolvePatternDeclaration(std::shared_ptr<ast::patterns::Pattern>,
  //                                 RibKind, const adt::CanonicalPath &prefix,
  //                                 const adt::CanonicalPath &canonicalPrefix);
  //  void resolvePatternDeclarationWithoutRange(
  //      std::shared_ptr<ast::patterns::PatternWithoutRange>, RibKind,
  //      std::vector<PatternBinding> &bindings, const adt::CanonicalPath
  //      &prefix, const adt::CanonicalPath &canonicalPrefix);
  //  void
  //  resolvePathPatternDeclaration(std::shared_ptr<ast::patterns::PathPattern>,
  //                                RibKind, std::vector<PatternBinding>
  //                                &bindings, const adt::CanonicalPath &prefix,
  //                                const adt::CanonicalPath &canonicalPrefix);
  //  void
  //  resolveTupleStructPatternDeclaration(std::shared_ptr<ast::patterns::TupleStructPattern>,
  //                                RibKind, std::vector<PatternBinding>
  //                                &bindings, const adt::CanonicalPath &prefix,
  //                                const adt::CanonicalPath &canonicalPrefix);
  //  void
  //  resolveStructPatternDeclaration(std::shared_ptr<ast::patterns::StructPattern>,
  //                                RibKind, std::vector<PatternBinding>
  //                                &bindings, const adt::CanonicalPath &prefix,
  //                                const adt::CanonicalPath &canonicalPrefix);

  // statements
  void resolveStatement(std::shared_ptr<ast::Statement>,
                        const adt::CanonicalPath &prefix,
                        const adt::CanonicalPath &canonicalPrefix,
                        const adt::CanonicalPath &empty);
  void resolveLetStatement(std::shared_ptr<ast::LetStatement>,
                           const adt::CanonicalPath &prefix,
                           const adt::CanonicalPath &canonicalPrefix);
  void resolveExpressionStatement(std::shared_ptr<ast::ExpressionStatement>,
                                  const adt::CanonicalPath &prefix,
                                  const adt::CanonicalPath &canonicalPrefix);
  void resolveStructStructStatement(std::shared_ptr<ast::StructStruct>,
                                    const adt::CanonicalPath &prefix,
                                    const adt::CanonicalPath &canonicalPrefix);
  void resolveTupleStructStatement(std::shared_ptr<ast::TupleStruct>,
                                   const adt::CanonicalPath &prefix,
                                   const adt::CanonicalPath &canonicalPrefix);

  void resolveConstParam(const ast::GenericParam &p,
                         const adt::CanonicalPath &prefix,
                         const adt::CanonicalPath &canonicalPrefix);
  void resolveTypeParam(const ast::GenericParam &p,
                        const adt::CanonicalPath &prefix,
                        const adt::CanonicalPath &canonicalPrefix);

  std::optional<basic::NodeId> resolveSimplePath(const ast::SimplePath &path);
  std::optional<basic::NodeId>
  resolvePathInExpression(std::shared_ptr<ast::PathInExpression>,
                          const adt::CanonicalPath &prefix,
                          const adt::CanonicalPath &canonicalPrefix);
  void resolveStructExpression(std::shared_ptr<ast::StructExpression> str,
                               const adt::CanonicalPath &prefix,
                               const adt::CanonicalPath &canonicalPrefix);
  void resolveStructExprFields(const ast::StructExprFields &fields,
                               const adt::CanonicalPath &prefix,
                               const adt::CanonicalPath &canonicalPrefix);

  void verifyAssignee(ast::Expression *);
  void verifyAssignee(ast::ExpressionWithoutBlock *);
  void verifyAssignee(ast::ExpressionWithBlock *);

  std::map<basic::NodeId, std::shared_ptr<ast::UseDeclaration>> useDeclarations;
  std::map<basic::NodeId, std::shared_ptr<ast::Module>> modules;

  void resolveAssociatedFunction(ast::Function *,
                                 const adt::CanonicalPath &prefix,
                                 const adt::CanonicalPath &canonicalPrefix);
  void resolveAssociatedTypeAlias(ast::TypeAlias *,
                                  const adt::CanonicalPath &prefix,
                                  const adt::CanonicalPath &canonicalPrefix);
  void resolveAssociatedConstantItem(ast::ConstantItem *,
                                     const adt::CanonicalPath &prefix,
                                     const adt::CanonicalPath &canonicalPrefix);
  void resolveAssociatedMacroInvocationSemi(
      ast::MacroInvocationSemiItem *, const adt::CanonicalPath &prefix,
      const adt::CanonicalPath &canonicalPrefix);
  void resolveAssociatedItemInTrait(const ast::AssociatedItem &,
                                    const adt::CanonicalPath &prefix,
                                    const adt::CanonicalPath &canonicalPrefix);
  void
  resolveMacroInvocationSemiInTrait(ast::MacroInvocationSemiItem *,
                                    const adt::CanonicalPath &prefix,
                                    const adt::CanonicalPath &canonicalPrefix);
  void resolveTypeAliasInTrait(ast::TypeAlias *,
                               const adt::CanonicalPath &prefix,
                               const adt::CanonicalPath &canonicalPrefix);
  void resolveConstantItemInTrait(ast::ConstantItem *,
                                  const adt::CanonicalPath &prefix,
                                  const adt::CanonicalPath &canonicalPrefix);
  void resolveFunctionInTrait(ast::Function *, const adt::CanonicalPath &prefix,
                              const adt::CanonicalPath &canonicalPrefix);

  std::vector<Import> determinedImports;

  //  std::map<adt::CanonicalPath,
  //           std::pair<std::shared_ptr<ast::Module>, basic::NodeId>>
  //      modules;

  void pushNewModuleScope(basic::NodeId moduleId) {
    currentModuleStack.push_back(moduleId);
  }

  void popModuleScope() { currentModuleStack.pop_back(); }

  basic::NodeId peekCurrentModuleScope() const {
    return currentModuleStack.back();
  }

  basic::NodeId peekParentModuleScope() const {
    assert(currentModuleStack.size() > 1);
    return currentModuleStack.at(currentModuleStack.size() - 2);
  }

  // void setUnitTypeNodeId(basic::NodeId id) { unitTyNodeId = id; }

  void insertResolvedName(basic::NodeId refId, basic::NodeId defId);
  void insertCapturedItem(basic::NodeId id);

  bool declNeedsCapture(basic::NodeId declRibNodeId,
                        basic::NodeId closureRibNodeId, const Scope &scope);

  tyctx::TyCtx *tyCtx;

  // types
  //  void generateBuiltins();
  //  void setupBuiltin(std::string_view name, type_checking::TyTy::BaseType
  //  *tyty);

  void insertBuiltinTypes(Rib *r);
  //  std::vector<std::pair<std::string, ast::types::TypeExpression *>> &
  //  getBuiltinTypes();

  // modules
  basic::NodeId peekCrateModuleScope() {
    assert(not currentModuleStack.empty());
    return currentModuleStack.front();
  }

  // Scopes
  Scope &getLabelScope() { return labelScope; }
  Scope &getMacroScope() { return macroScope; }

  Scope nameScope;
  Scope typeScope;
  Scope labelScope;
  Scope macroScope;

  // Ribs
  void pushNewNameRib(Rib *);
  void pushNewTypeRib(Rib *);
  void pushNewLabelRib(Rib *);
  void pushNewMaroRib(Rib *);

  // map a node to a rib
  std::map<basic::NodeId, Rib *> nameRibs;
  std::map<basic::NodeId, Rib *> typeRibs;
  std::map<basic::NodeId, Rib *> labelRibs;
  std::map<basic::NodeId, Rib *> macroRibs;

  // keep track of the current module scope ids
  std::vector<basic::NodeId> currentModuleStack;

  // captured variables by current closure
  std::vector<basic::NodeId> closureContext;
  std::map<basic::NodeId, std::set<basic::NodeId>> closureCaptureMappings;

  // resolved items: reference -> definition
  std::map<basic::NodeId, basic::NodeId> resolvedNames;
  std::map<basic::NodeId, basic::NodeId> resolvedTypes;
  std::map<basic::NodeId, basic::NodeId> resolvedLabels;
  std::map<basic::NodeId, basic::NodeId> resolvedMacros;
  std::map<basic::NodeId, basic::NodeId> miscResolvedItems;

  // closures
  void pushClosureContext(basic::NodeId);
  void popClosureContext();
};

} // namespace rust_compiler::sema::resolver

// FIXME: Scoped
// FIXME: store canonical paths
