#include "Mangler/Mangler.h"

#include "AST/Crate.h"
#include "AST/Function.h"
#include "AST/GenericArgsConst.h"
#include "AST/Implementation.h"
#include "AST/InherentImpl.h"
#include "AST/Module.h"
#include "AST/PathIdentSegment.h"
#include "AST/Struct.h"
#include "AST/StructStruct.h"
#include "AST/TraitImpl.h"
#include "AST/Types/ArrayType.h"
#include "AST/Types/BareFunctionType.h"
#include "AST/Types/ReferenceType.h"
#include "AST/Types/SliceType.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypeNoBounds.h"
#include "AST/Types/TypePath.h"
#include "AST/Types/TypePathSegment.h"
#include "AST/VisItem.h"
#include "Lexer/Identifier.h"
#include "TyCtx/TyTy.h"

#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <optional>
#include <sstream>
#include <string>

using namespace rust_compiler::ast;
using namespace rust_compiler::ast::types;

namespace rust_compiler::mangler {

// std::string
// Mangler::mangleFreestandingFunction(std::span<const ast::VisItem *> path,
//                                     ast::Crate *crate) {}
//
// std::string Mangler::mangleMethod(std::span<const ast::VisItem *> path,
//                                   ast::Crate *crate) {}
//
// std::string Mangler::mangleStatic(std::span<const ast::VisItem *> path,
//                                   ast::Crate *crate) {}
//
// std::string Mangler::mangleClosure(std::span<const ast::VisItem *> path,
//                                    ast::Crate *crate,
//                                    ast::ClosureExpression *closure) {}

constexpr llvm::StringLiteral
    CODES("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");

static std::string toBase62(int value) {
  std::string str;

  do {
    str.insert(0, std::string(1, CODES[value % 62]));
    value /= 62;
  } while (value > 0);

  return str;
}

std::string Mangler::mangle(std::span<const ast::VisItem *> path,
                            ast::Crate *crate) {
  std::vector<std::string> tags;
  std::string symbolPath;
  llvm::raw_string_ostream mangledName(symbolPath);

  // _R0 FIXME

  // tags

  // instantiating-crate
  mangledName << "C" << std::to_string(crate->getCrateName().size())
              << crate->getCrateName();

  for (const ast::VisItem *it : path) {
    switch (it->getKind()) {
    case VisItemKind::Module: {
      const ast::Module *module = static_cast<const Module *>(it);
      Identifier name = module->getModuleName();
      mangledName << mangleIdentifier(name);
      tags.push_back("Nt");
      break;
    }
    case VisItemKind::ExternCrate: {
      break;
    }
    case VisItemKind::UseDeclaration: {
      llvm_unreachable("we don't mangle use declarations");
      break;
    }
    case VisItemKind::Function: {
      const Function *fun = static_cast<const Function *>(it);
      Identifier name = fun->getName();
      mangledName << mangleIdentifier(name);
      tags.push_back("Nv");
      break;
    }
    case VisItemKind::TypeAlias: {
      break;
    }
    case VisItemKind::Struct: {
      const Struct *stru = static_cast<const Struct *>(it);
      switch (stru->getKind()) {
      case StructKind::StructStruct2: {
        const StructStruct *struc2 = static_cast<const StructStruct *>(stru);
        Identifier name = struc2->getIdentifier();
        mangledName << mangleIdentifier(name);
        tags.push_back("Nu");
        break;
      }
      case StructKind::TupleStruct2: {
        break;
      }
      }
      break;
    }
    case VisItemKind::Enumeration: {
      break;
    }
    case VisItemKind::Union: {
      break;
    }
    case VisItemKind::ConstantItem: {
      break;
    }
    case VisItemKind::StaticItem: {
      break;
    }
    case VisItemKind::Trait: {
      break;
    }
    case VisItemKind::Implementation: {
      const ast::Implementation *implementation =
          static_cast<const ast::Implementation *>(it);
      switch (implementation->getKind()) {
      case ImplementationKind::InherentImpl: {
        const ast::InherentImpl *impl =
            static_cast<const ast::InherentImpl *>(it);
        mangledName << mangleType(impl->getType().get());
        tags.push_back("M");
        break;
      }
      case ImplementationKind::TraitImpl: {
        const ast::TraitImpl *impl = static_cast<const ast::TraitImpl *>(it);
        mangledName << mangleType(impl->getType().get());
        mangledName << mangleType(impl->getTypePath().get());
        tags.push_back("X");
        break;
      }
      }
      break;
    }
    case VisItemKind::ExternBlock: {
      break;
    }
    }
  }

  std::reverse(tags.begin(), tags.end());

  // _R0 FIXME

  // tags

  std::string result;

  llvm::raw_string_ostream mangled(result);

  mangled << "_R0";
  for (const std::string &tag : tags)
    mangled << tag;

  mangled << mangledName.str();

  // vendor-specific-suffix
  mangled << "$rust_compiler";

  return mangled.str();
}

std::string Mangler::mangleType(const ast::types::TypeExpression *type) {
  switch (type->getKind()) {
  case TypeExpressionKind::TypeNoBounds: {
    const TypeNoBounds *noBounds = static_cast<const TypeNoBounds *>(type);
    switch (noBounds->getKind()) {
    case TypeNoBoundsKind::ParenthesizedType:
      break;
    case TypeNoBoundsKind::ImplTraitType:
      break;
    case TypeNoBoundsKind::ImplTraitTypeOneBound:
      break;
    case TypeNoBoundsKind::TraitObjectTypeOneBound:
      // FIXME: dyn
      break;
    case TypeNoBoundsKind::TypePath: {
      const TypePath *path = static_cast<const TypePath *>(type);
      return mangleTypePath(path);
    }
    case TypeNoBoundsKind::TupleType: {
      const TupleType *tuple = static_cast<const TupleType *>(type);
      return mangleTupleType(tuple);
    }
    case TypeNoBoundsKind::NeverType: {
      const TypePath *path = static_cast<const TypePath *>(type);
      return mangleTypePath(path);
    }
    case TypeNoBoundsKind::RawPointerType: {
      const RawPointerType *pointer = static_cast<const RawPointerType *>(type);
      return mangleRawPointerType(pointer);
    }
    case TypeNoBoundsKind::ReferenceType: {
      const ReferenceType *refer = static_cast<const ReferenceType *>(type);
      return mangleReferenceType(refer);
    }
    case TypeNoBoundsKind::ArrayType: {
      const ArrayType *array = static_cast<const ArrayType *>(type);
      std::string result;
      uint64_t N = evaluator.foldAsUsize(array->getExpression().get());
      llvm::raw_string_ostream mangled(result);
      mangled << "A" << mangleType(array->getType().get()) << mangleConst(N);
      return mangled.str();
    }
    case TypeNoBoundsKind::SliceType: {
      const SliceType *slice = static_cast<const SliceType *>(type);
      std::string result;
      llvm::raw_string_ostream mangled(result);
      mangled << "S" << mangleType(slice->getType().get());
      return mangled.str();
    }
    case TypeNoBoundsKind::InferredType:
      break;
    case TypeNoBoundsKind::QualifiedPathInType: {
      // FIXME path
      break;
    }
    case TypeNoBoundsKind::BareFunctionType: {
      return mangleBareFunctionType(
          static_cast<const ast::types::BareFunctionType *>(type));
    }
    case TypeNoBoundsKind::MacroInvocation:
      break;
    }
    break;
  }
  case TypeExpressionKind::ImplTraitType: {
    break;
  }
  case TypeExpressionKind::TraitObjectType: {
    // FIXME: dyn
    break;
  }
  }

  return mangleBackref();
}

std::string Mangler::mangleTypePath(const ast::types::TypePath *path) {
  if (path->getNrOfSegments() == 1) {
    std::optional<std::string> maybeBasicType =
        tryBasicType(path->getSegments()[0]);
    if (maybeBasicType)
      return *maybeBasicType;
  }
  std::string result;
  llvm::raw_string_ostream mangled(result);
  // mangled << "F" /*<< mangleType(slice->getType().get())*/;
  for (const TypePathSegment &seg : path->getSegments()) {
    mangled << manglePathIdentSegment(seg.getSegment());
    if (seg.hasGenerics())
      mangled << mangleGenericArgs(seg.getGenericArgs());
    if (seg.hasTypeFunction())
      mangled << mangleTypePathFunction(seg.getTypePathFn());
  }
  return mangled.str();
}

std::string Mangler::mangleConst(uint64_t c) const {
  std::stringstream stream;
  stream << std::hex << c << "_";
  std::string result(stream.str());
  return result;
}

std::string
Mangler::mangleBareFunctionType(const ast::types::BareFunctionType *funType) {
  std::string result;
  llvm::raw_string_ostream mangled(result);
  mangled << "F" /*<< mangleType(slice->getType().get())*/;

  FunctionTypeQualifiers quals = funType->getQualifiers();
  if (quals.isUnsafe())
    mangled << "U";
  if (quals.hasAbi())
    mangled << "K" << quals.getAbi().getAbi();

  if (funType->hasParameters()) {
    std::shared_ptr<FunctionParametersMaybeNamedVariadic> params =
        funType->getParameters();
    switch (params->getKind()) {
    case FunctionParametersMaybeNamedVariadicKind::
        MaybeNamedFunctionParameters: {
      auto par = std::static_pointer_cast<MaybeNamedFunctionParameters>(params);
      break;
    }
    case FunctionParametersMaybeNamedVariadicKind::
        MaybeNamedFunctionParametersVariadic: {
      auto par = std::static_pointer_cast<MaybeNamedFunctionParametersVariadic>(
          params);
      break;
    }
    }
  }

  if (funType->hasReturnType()) {
    BareFunctionReturnType ret = funType->getReturnType();
    mangled << "E" << mangleType(ret.getType().get());
  }
  return mangled.str();
}

std::optional<std::string>
Mangler::tryBasicType(const ast::types::TypePathSegment &seg) const {
  if (seg.hasGenerics())
    return std::nullopt;
  if (seg.hasTypeFunction())
    return std::nullopt;
  PathIdentSegment ident = seg.getSegment();
  if (ident.getKind() != PathIdentSegmentKind::Identifier)
    return std::nullopt;
  std::string id = ident.getIdentifier().toString();

  if (id == "i8")
    return "a";
  else if (id == "bool")
    return "b";
  else if (id == "char")
    return "c";
  else if (id == "f64")
    return "d";
  else if (id == "str")
    return "e";
  else if (id == "f32")
    return "f";
  else if (id == "u8")
    return "h";
  else if (id == "isize")
    return "i";
  else if (id == "usize")
    return "j";
  else if (id == "i32")
    return "l";
  else if (id == "u32")
    return "m";
  else if (id == "i128")
    return "n";
  else if (id == "u128")
    return "o";
  else if (id == "i16")
    return "s";
  else if (id == "u16")
    return "t";
  else if (id == "()")
    return "u";
  else if (id == "...")
    return "v";
  else if (id == "i64")
    return "x";
  else if (id == "u64")
    return "y";
  else if (id == "!")
    return "z";
  else if (id == "_")
    return "p";

  return std::nullopt;
}

std::string Mangler::mangleTupleType(const ast::types::TupleType *tuple) {
  std::string result;
  llvm::raw_string_ostream mangled(result);
  mangled << "T";
  for (auto &type : tuple->getTypes())
    mangled << mangleType(type.get());
  mangled << "E";

  return mangled.str();
}

std::string
Mangler::mangleRawPointerType(const ast::types::RawPointerType *pointer) {
  std::string result;
  llvm::raw_string_ostream mangled(result);
  if (pointer->isMut())
    mangled << "O";
  else if (pointer->isConst())
    mangled << "P";

  mangled << mangleType(pointer->getType().get());

  return mangled.str();
}

std::string
Mangler::mangleReferenceType(const ast::types::ReferenceType *refer) {
  std::string result;
  llvm::raw_string_ostream mangled(result);

  if (refer->isMutable())
    mangled << "Q";
  else
    mangled << "R";

  if (refer->hasLifetime())
    mangled << mangleLifetime(refer->getLifetime());

  mangled << mangleType(refer->getReferencedType().get());

  return mangled.str();
}

std::string Mangler::mangleLifetime(const ast::Lifetime &l) const {
  std::string result;
  llvm::raw_string_ostream mangled(result);

  mangled << "L";

  // FIXME XX

  return mangled.str();
}

std::string Mangler::mangleBackref() const {
  static int backCounter = 10;
  return toBase62(backCounter++);
}

std::string
Mangler::manglePathIdentSegment(const ast::PathIdentSegment &segment) const {
  if (segment.getKind() == PathIdentSegmentKind::Identifier)
    return mangleIdentifier(segment.getIdentifier());

  return segment.toString();
}

std::string Mangler::mangleGenericArgs(const ast::GenericArgs &args) {
  std::string result;
  llvm::raw_string_ostream mangled(result);

  for (const auto &arg : args.getArgs())
    mangled << mangleGenericArg(arg);

  return mangled.str();
}

std::string
Mangler::mangleTypePathFunction(const ast::types::TypePathFn &) const {
  // FIXME: unsupported?
  return "";
}

std::string Mangler::mangleGenericArg(const ast::GenericArg &arg) {
  switch (arg.getKind()) {
  case GenericArgKind::Lifetime: {
    return mangleLifetime(arg.getLifetime());
  }
  case GenericArgKind::Type: {
    return mangleType(arg.getType().get());
  }
  case GenericArgKind::Const: {
    std::string result;
    llvm::raw_string_ostream mangled(result);
    mangled << "K" << mangleGenericArgsConst(arg.getConst());
    return mangled.str();
  }
  case GenericArgKind::Binding: {
    // FIXME: unsupported?
    return "";
  }
  }
}

std::string
Mangler::mangleGenericArgsConst(const ast::GenericArgsConst &cons) const {
  switch (cons.getKind()) {
  case GenericArgsConstKind::BlockExpression: {
    break;
  }
  case GenericArgsConstKind::LiteralExpression: {
    break;
  }
  case GenericArgsConstKind::SimplePathSegment: {
    break;
  }
  }
  // FIXME: how?
  return "";
}

std::string Mangler::mangleIdentifier(const Identifier &ident) const {
  static int disambiguator = 10;

  std::vector<uint8_t> bytes = ident.getAsBytes();

  std::string result;
  llvm::raw_string_ostream mangled(result);

  // FIXME
  mangled << "s" << toBase62(disambiguator++) << "u"
          << std::to_string(bytes.size()) << "_";

  for (uint8_t byte : bytes)
    mangled << byte;

  return mangled.str();
}

} // namespace rust_compiler::mangler
