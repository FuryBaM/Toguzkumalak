#pragma once
#include "train.h"
#include "config.h"

void combine_datasets(const std::string& folder, const std::string& output_file);
void run_arena(Config& config);
