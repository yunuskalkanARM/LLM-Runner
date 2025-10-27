//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#include "Logger.hpp"

#include <cstdarg>
#include <vector>
#include <stdexcept>
#include <cstdio>

namespace LlmLog {

    std::string vformat_va(const char* format, va_list args) {
        va_list tmp;
        va_copy(tmp, args);
        int n = std::vsnprintf(nullptr, 0, format, tmp);
        va_end(tmp);

        if (n < 0) {
            va_end(args);
            throw std::runtime_error("Error while logging a message: logging string sizing failed.\n");
        }

        std::vector<char> buf(static_cast<size_t>(n) + 1);
        const int n2 = std::vsnprintf(buf.data(), buf.size(), format, args);
        (void)n2;

        return buf.data();

    }
} // namespace LlmLog
