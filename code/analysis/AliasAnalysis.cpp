#include "Analysis/AliasAnalysis.h"

#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace rust_compiler::analysis {

using namespace mlir;

std::optional<AliasResult> mayAlias(mlir::AliasAnalysis *alias,
                                    mlir::Operation *a, mlir::Operation *b) {
  MemoryEffectOpInterface interfaceA =
      dyn_cast<mlir::MemoryEffectOpInterface>(a);
  if (!interfaceA)
    return std::nullopt;
  MemoryEffectOpInterface interfaceB =
      dyn_cast<mlir::MemoryEffectOpInterface>(b);
  if (!interfaceB)
    return std::nullopt;

  // Build a ModRefResult by merging the behavior of the effects of this
  // operation.
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance> effectsA;
  interfaceA.getEffects(effectsA);
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance> effectsB;
  interfaceB.getEffects(effectsB);

  AliasResult result = AliasResult::MayAlias;

  for (const mlir::MemoryEffects::EffectInstance &effectA : effectsA) {
    if (mlir::Value effectValueA = effectA.getValue()) {
      for (const mlir::MemoryEffects::EffectInstance &effectB : effectsB) {
        if (mlir::Value effectValueB = effectB.getValue()) {
          AliasResult tmp = alias->alias(effectValueA, effectValueB);
          // If we don't alias, ignore this effect.
          if (tmp.isNo())
            continue;

          result = result.merge(tmp);
        }
      }
    }
  }

  return result;
}

} // namespace rust_compiler::analysis

//  https://reviews.llvm.org/D136889

// ModRefResult getModRef(Operation *op, Value location) {
//   MemoryEffectOpInterface interface = dyn_cast<MemoryEffectOpInterface>(op);
//   if (!interface)
//     return ModRefResult::getModAndRef();
//
//   // Build a ModRefResult by merging the behavior of the effects of this
//   // operation.
//   SmallVector<MemoryEffects::EffectInstance> effects;
//   interface.getEffects(effects);
//
//   ModRefResult result = ModRefResult::getNoModRef();
//   for (const MemoryEffects::EffectInstance &effect : effects) {
//
//     // Check for an alias between the effect and our memory location.
//     AliasResult aliasResult = AliasResult::MayAlias;
//     if (Value effectValue = effect.getValue())
//       aliasResult = alias(effectValue, location);
//
//     // If we don't alias, ignore this effect.
//     if (aliasResult.isNo())
//       continue;
//
//     // Merge in the corresponding mod or ref for this effect.
//     if (isa<MemoryEffects::Read>(effect.getEffect())) {
//       result = result.merge(ModRefResult::getRef());
//     } else {
//       result = result.merge(ModRefResult::getMod());
//     }
//     if (result.isModAndRef())
//       break;
//   }
//   return result;
// }
