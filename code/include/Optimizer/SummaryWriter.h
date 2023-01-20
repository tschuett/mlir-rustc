#pragma once


#include <string>

#include "Optimizer/Passes.h.inc"

#define GEN_PASS_DECL_SUMMARYWRITERPASS

struct SummaryWriterPassOptions {
  std::string file = ""
};


#undef GEN_PASS_DECL_SUMMARYWRITERPASS
