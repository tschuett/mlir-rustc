#pragma once

#include "ADT/CanonicalPath.h"
#include "ADT/ScopedCanonicalPath.h"
#include "AST/Crate.h"
#include "AST/Implementation.h"
#include "AST/InherentImpl.h"
#include "AST/MacroItem.h"
#include "AST/TraitImpl.h"
#include "AST/UseDeclaration.h"
#include "AST/VisItem.h"
#include "Basic/Ids.h"

#include <map>
#include <stack>
#include <string_view>
#include <vector>

namespace rust_compiler::sema::resolver {

/// https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/late/struct.Rib.html
class Rib {
public:
  // https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/late/enum.RibKind.html
  enum class RibKind { Type };

  Rib(RibKind kind) : kind(kind) {}

private:
  std::map<std::string, basic::NodeId> bindings;
  RibKind kind;
};

class Scope {
public:
  Scope(basic::CrateNum crateNum);

  Rib *peek();
  void push(basic::NodeId id);

  basic::CrateNum getCrateNum() const { return crateNum; }

private:
  basic::CrateNum crateNum;
  basic::NodeId node_id;
  std::vector<Rib *> stack;
};

class Segment {
  std::string name;
};

class Import {
public:
  enum class ImportKind { Single, Glob, ExternCrate, MacroUse, MacroExport };

private:
  ImportKind kind;
  basic::NodeId nodeId;
  llvm::SmallVector<Segment> modulePath;
};

class Resolver {
public:
  Resolver() = delete;
  Resolver(std::string_view crateName)
      : scopedPath(adt::CanonicalPath::newSegment(getNextNodeId(), "crate",
                                                  crateName)) {}

  ~Resolver() = default;

  void resolveCrate(std::shared_ptr<ast::Crate>);

private:
  adt::ScopedCanonicalPath scopedPath;

  void resolveVisItem(std::shared_ptr<ast::VisItem>);
  void resolveMacroItem(std::shared_ptr<ast::MacroItem>);
  void resolveImplementation(std::shared_ptr<ast::Implementation>);
  void resolveUseDeclaration(std::shared_ptr<ast::UseDeclaration>);

  void resolveInherentImpl(std::shared_ptr<ast::InherentImpl>);
  void resolveTraitImpl(std::shared_ptr<ast::TraitImpl>);

  std::map<basic::NodeId, std::shared_ptr<ast::UseDeclaration>> useDeclarations;
  std::map<basic::NodeId, std::shared_ptr<ast::Module>> modules;

  basic::NodeId nodeId = 0;

  basic::NodeId getNextNodeId();
  void addModule(std::shared_ptr<ast::Module> mod, basic::NodeId,
                 const adt::CanonicalPath &path);

  std::vector<Import> determinedImports;
};

} // namespace rust_compiler::sema::resolver

// FIXME: Scoped
