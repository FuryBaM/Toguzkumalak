#pragma once

#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <bit>
#include <memory>
#include <stdexcept>
#include <limits>
#include <vector>
#include <array>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <random>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <filesystem>

#ifndef _WIN32
#include <cstring>
#include <sched.h>
#include <pthread.h>
#endif

#ifdef _BUILD_MCTS
#include <torch/torch.h>
#include <torch/script.h>
#include <omp.h>
#include <future>
#endif

