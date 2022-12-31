#include "Mir/MirDialect.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Types.h>
#include <mlir/Transforms/InliningUtils.h>
#include <optional>

#include <llvm/Support/Debug.h>
#include <llvm/Support/WithColor.h>

#define DEBUG_TYPE "MirDialect"

using namespace mlir;
using namespace rust_compiler::Mir;

#include "Mir/MirDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Mir/MirOps.cpp.inc"

void MirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Mir/MirOps.cpp.inc"
      >();
}

#include "Mir/MirOps.cpp.inc"

namespace rust_compiler::Mir {

std::optional<llvm::StringRef>
MirDialect::findMethod(mlir::TypeID BaseType,
                       llvm::StringRef MethodName) const {
  return methods.lookupMethod(BaseType, MethodName);
}

std::optional<llvm::StringRef>
MirDialect::findMethodFromBaseClass(mlir::TypeID BaseType,
                                    llvm::StringRef MethodName) const {
  if (auto Method = findMethod(BaseType, MethodName))
    return Method;
  for (auto DerivedTy : getDerivedTypes(BaseType)) {
    if (auto Method = findMethod(DerivedTy, MethodName))
      return Method;
  }
  return std::nullopt;
}

void MirDialect::registerMethodDefinition(llvm::StringRef Name,
                                          mlir::func::FuncOp Func) {
  methods.registerDefinition(Name, Func);
}

std::optional<mlir::func::FuncOp>
MirDialect::lookupMethodDefinition(llvm::StringRef Name,
                                   mlir::FunctionType Type) const {
  return methods.lookupDefinition(Name, Type);
}

template <typename T> static void addMirMethod(MethodRegistry &methods) {
  if constexpr (Mir::isMirMethod<T>::value) {
    // If the operation is a Mir method, register it.
    const auto TypeID = T::getTypeID();
    const llvm::StringRef OpName = T::getOperationName();
    for (llvm::StringRef Name : T::getMethodNames())
      assert(methods.registerMethod(TypeID, Name, OpName) &&
             "Duplicated method");
  }
}

template <typename... Args> void MirDialect::addOperations() {
  mlir::Dialect::addOperations<Args...>();
  (addMirMethod<Args>(methods), ...);
}

void MethodRegistry::init(mlir::MLIRContext &Ctx) {
  assert(!Module && "Registry already initialized");
  Module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&Ctx), ModuleName);
}

std::optional<llvm::StringRef>
MethodRegistry::lookupMethod(mlir::TypeID BaseType,
                             llvm::StringRef MethodName) const {
  const auto Iter = Methods.find({BaseType, MethodName});
  return Iter == Methods.end() ? std::nullopt
                               : std::optional<llvm::StringRef>{Iter->second};
}

bool MethodRegistry::registerMethod(mlir::TypeID TypeID,
                                    llvm::StringRef MethodName,
                                    llvm::StringRef OpName) {
  return Methods.try_emplace({TypeID, MethodName}, OpName).second;
}

void MethodRegistry::registerDefinition(llvm::StringRef Name,
                                        mlir::func::FuncOp Func) {
  LLVM_DEBUG(llvm::dbgs() << "Registering function \"" << Name << "\": " << Func
                          << "\n");
  auto Clone = Func.clone();
  const auto FuncType = Clone.getFunctionType();
  auto Iter =
      Definitions.insert_as<std::pair<llvm::StringRef, mlir::FunctionType>>(
          {{Name, FuncType}, Clone}, {Name, FuncType});
  if (!Iter.second) {
    // Override current function.
    auto &ToOverride = Iter.first->second;
    assert(ToOverride.isDeclaration() && "Only a declaration can be overriden");
    assert(!Func.isDeclaration() &&
           "A declaration cannot be used to override another declaration");
    assert(ToOverride.getName() == Func.getName() &&
           "Functions must have the same mangled name");
    ToOverride.erase();
    ToOverride = Clone;
  }
  Module.push_back(Clone);
}

std::optional<mlir::func::FuncOp> MethodRegistry::lookupDefinition(
    llvm::StringRef Name, mlir::FunctionType FuncType) const {
  LLVM_DEBUG(llvm::dbgs() << "Fetching function \"" << Name
                          << "\" with type: " << FuncType << "\n");

  const auto Iter =
      Definitions.find_as<std::pair<llvm::StringRef, mlir::FunctionType>>(
          {Name, FuncType});
  if (Iter == Definitions.end()) {
    llvm::WithColor::warning() << "Could not find function \"" << Name
                               << "\" with type " << FuncType << "\n";
    return std::nullopt;
  }
  LLVM_DEBUG(llvm::dbgs() << "Function found: " << Iter->second << "\n");
  return Iter->second;
}

} // namespace rust_compiler::Mir
