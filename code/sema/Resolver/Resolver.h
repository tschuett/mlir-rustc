#pragma once

#include "ADT/CanonicalPath.h"
#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/ClosureExpression.h"
#include "AST/Crate.h"
#include "AST/Expression.h"
#include "AST/ExpressionStatement.h"
#include "AST/Implementation.h"
#include "AST/InfiniteLoopExpression.h"
#include "AST/InherentImpl.h"
#include "AST/LetStatement.h"
#include "AST/LoopExpression.h"
#include "AST/MacroItem.h"
#include "AST/MatchExpression.h"
#include "AST/OperatorExpression.h"
#include "AST/PathExpression.h"
#include "AST/PathInExpression.h"
#include "AST/Patterns/PathPattern.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Patterns/PatternWithoutRange.h"
#include "AST/QualifiedPathInExpression.h"
#include "AST/ReturnExpression.h"
#include "AST/SimplePath.h"
#include "AST/Statement.h"
#include "AST/StaticItem.h"
#include "AST/Struct.h"
#include "AST/StructStruct.h"
#include "AST/Trait.h"
#include "AST/TraitImpl.h"
#include "AST/TupleStruct.h"
#include "AST/Types/TupleType.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypePathFn.h"
#include "AST/UseDeclaration.h"
#include "AST/VisItem.h"
#include "AST/IfExpression.h"
#include "AST/Visiblity.h"
#include "Basic/Ids.h"
#include "Location.h"
#include "TyCtx/TyCtx.h"
#include "AST/ComparisonExpression.h"
#include "AST/DereferenceExpression.h"
#include "AST/BorrowExpression.h"

//#include "../TypeChecking/TyTy.h"
// #include "../TypeChecking/TypeChecking.h"

#include <cassert>
#include <map>
#include <optional>
#include <set>
#include <stack>
#include <string_view>
#include <vector>

namespace rust_compiler::sema::resolver {

// Specifies whether the set of already bound patterns are related by 'Or' or
// 'Product'. Used to check for multiple bindings to the same identifier.
enum class PatternBoundCtx {
  // A product pattern context (e.g. struct and tuple patterns)
  Product,
  // An or-pattern context (e.g. p_0 | p_1 | ...)
  Or,
};

struct PatternBinding {
  PatternBoundCtx ctx;
  std::set<std::string> idents;

  PatternBinding(PatternBoundCtx ctx, std::set<std::string> idents)
      : ctx(ctx), idents(idents) {}
};

/// https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/late/index.html
/// https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/index.html
///  https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/struct.Resolver.html

/// https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/late/struct.Rib.html
enum class RibKind { Dummy, Function, Label, Parameter, Type, Variable };

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

  void insert(const adt::CanonicalPath &, basic::NodeId, Location, RibKind);
  void appendReferenceForDef(basic::NodeId ref, basic::NodeId def);

  bool wasDeclDeclaredInCurrentScope(basic::NodeId def) const;
  std::optional<basic::NodeId> lookup(const adt::CanonicalPath &p);

  std::optional<RibKind> lookupDeclType(basic::NodeId id);
  std::optional<Rib *> lookupRibForDecl(basic::NodeId id);

  const std::vector<Rib *> &getContext() const { return stack; };

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

private:
  // items no recurse
  void resolveVisItemNoRecurse(std::shared_ptr<ast::VisItem>,
                               const adt::CanonicalPath &prefix,
                               const adt::CanonicalPath &canonicalPrefix);
  void resolveMacroItemNoRecurse(std::shared_ptr<ast::MacroItem>,
                                 const adt::CanonicalPath &prefix,
                                 const adt::CanonicalPath &canonicalPrefix);
  void resolveFunctionNoRecurse(std::shared_ptr<ast::Function>,
                                const adt::CanonicalPath &prefix,
                                const adt::CanonicalPath &canonicalPrefix);

  // items
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
  void resolveFunction(std::shared_ptr<ast::Function>,
                       const adt::CanonicalPath &prefix,
                       const adt::CanonicalPath &canonicalPrefix);
  void resolveModule(std::shared_ptr<ast::Module>,
                     const adt::CanonicalPath &prefix,
                     const adt::CanonicalPath &canonicalPrefix);
  void resolveStructItem(std::shared_ptr<ast::Struct>,
                         const adt::CanonicalPath &prefix,
                         const adt::CanonicalPath &canonicalPrefix);
  void resolveStructStructItem(std::shared_ptr<rust_compiler::ast::StructStruct>,
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

  // expressions
  void resolveExpression(std::shared_ptr<ast::Expression>,
                         const adt::CanonicalPath &prefix,
                         const adt::CanonicalPath &canonicalPrefix);
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
  void resolveOperatorExpression(std::shared_ptr<ast::OperatorExpression>,
                                 const adt::CanonicalPath &prefix,
                                 const adt::CanonicalPath &canonicalPrefix);
  void resolveLoopExpression(std::shared_ptr<ast::LoopExpression>,
                             const adt::CanonicalPath &prefix,
                             const adt::CanonicalPath &canonicalPrefix);
  void resolveIfExpression(std::shared_ptr<ast::IfExpression>,
                             const adt::CanonicalPath &prefix,
                             const adt::CanonicalPath &canonicalPrefix);
  void resolvePathExpression(std::shared_ptr<ast::PathExpression>);
  void resolveBlockExpression(std::shared_ptr<ast::BlockExpression>,
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
  void resolveQualifiedPathInExpression(
      std::shared_ptr<ast::QualifiedPathInExpression>);
  void resolveArithmeticOrLogicalExpression(
      std::shared_ptr<ast::ArithmeticOrLogicalExpression>,
      const adt::CanonicalPath &prefix,
      const adt::CanonicalPath &canonicalPrefix);
  void resolveComparisonExpression(std::shared_ptr<ast::ComparisonExpression>,
                                const adt::CanonicalPath &prefix,
                                const adt::CanonicalPath &canonicalPrefix);
  void resolveDereferenceExpression(std::shared_ptr<ast::DereferenceExpression>,
                                const adt::CanonicalPath &prefix,
                                const adt::CanonicalPath &canonicalPrefix);
  void resolveBorrowExpression(std::shared_ptr<ast::BorrowExpression>,
                                const adt::CanonicalPath &prefix,
                                const adt::CanonicalPath &canonicalPrefix);

  // types
  std::optional<basic::NodeId>
      resolveType(std::shared_ptr<ast::types::TypeExpression>);
  std::optional<basic::NodeId>
      resolveTypeNoBounds(std::shared_ptr<ast::types::TypeNoBounds>);
  std::optional<basic::NodeId>
      resolveRelativeTypePath(std::shared_ptr<ast::types::TypePath>);
  void resolveTypePathFunction(const ast::types::TypePathFn &);

  // checks
  void resolveVisibility(std::optional<ast::Visibility>);

  // generics
  void resolveWhereClause(const ast::WhereClause &);
  void resolveGenericParams(const ast::GenericParams &,
                            const adt::CanonicalPath &prefix,
                            const adt::CanonicalPath &canonicalPrefix);
  void resolveGenericArgs(const ast::GenericArgs &);

  // patterns
  void resolvePatternDeclarationWithBindings(
      std::shared_ptr<ast::patterns::PatternNoTopAlt>, RibKind,
      std::vector<PatternBinding> &bindings);
  void
      resolvePatternDeclaration(std::shared_ptr<ast::patterns::PatternNoTopAlt>,
                                RibKind);
  void resolvePatternDeclarationWithoutRange(
      std::shared_ptr<ast::patterns::PatternWithoutRange>, RibKind,
      std::vector<PatternBinding> &bindings);
  void
      resolvePathPatternDeclaration(std::shared_ptr<ast::patterns::PathPattern>,
                                    RibKind);

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

  std::optional<basic::NodeId> resolveSimplePath(const ast::SimplePath &path);
  std::optional<basic::NodeId>
      resolvePathInExpression(std::shared_ptr<ast::PathInExpression>);

  void verifyAssignee(std::shared_ptr<ast::Expression>);
  std::map<basic::NodeId, std::shared_ptr<ast::UseDeclaration>> useDeclarations;
  std::map<basic::NodeId, std::shared_ptr<ast::Module>> modules;

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

  //void setUnitTypeNodeId(basic::NodeId id) { unitTyNodeId = id; }

  void insertResolvedName(basic::NodeId refId, basic::NodeId defId);
  void insertResolvedType(basic::NodeId refId, basic::NodeId defId);
  void insertCapturedItem(basic::NodeId id);

  bool declNeedsCapture(basic::NodeId declRibNodeId,
                        basic::NodeId closureRibNodeId, const Scope &scope);

  tyctx::TyCtx *tyCtx;

  // types
//  void generateBuiltins();
//  void setupBuiltin(std::string_view name, type_checking::TyTy::BaseType *tyty);

  void insertBuiltinTypes(Rib *r);
//  std::vector<std::pair<std::string, ast::types::TypeExpression *>> &
//  getBuiltinTypes();

  // modules
  basic::NodeId peekCrateModuleScope() {
    assert(not currentModuleStack.empty());
    return currentModuleStack.front();
  }

  // Scopes
  Scope &getTypeScope() { return typeScope; }
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
};

} // namespace rust_compiler::sema::resolver

// FIXME: Scoped
// FIXME: store canonical paths
