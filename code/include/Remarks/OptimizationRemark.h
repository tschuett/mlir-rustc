#pragma once

#include <string>

namespace rust_compiler::remarks {

class OptimizationRemarkBase {
  std::string functionName;
public:
  virtual ~OptimizationRemarkBase() = default;

  virtual OptimizationRemarkBase operator()();
};

class OptimizationRemark : public OptimizationRemarkBase {

public:
  virtual ~OptimizationRemark() = default;
};

class OptimizationRemarkMissed : public OptimizationRemarkBase {
public:
  virtual ~OptimizationRemarkMissed() = default;
};

class OptimizationRemarkAnalysis : public OptimizationRemarkBase {
public:
  virtual ~OptimizationRemarkAnalysis() = default;
};

} // namespace rust_compiler::remarks
