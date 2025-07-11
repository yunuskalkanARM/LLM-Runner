//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#ifndef LLM_TEST_UTILS_HPP
#define LLM_TEST_UTILS_HPP

#include "LlmConfig.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace Llm::Test::Utils {

std::unordered_map<std::string, std::string> LoadConfig(const std::string& configFilePath);

/**
 * Method to load LLM-Config params from file in "param=value(int)" format
 * @param userConfigFilePath
 * @return a dictionary of user-defined params.
 */
std::unordered_map<std::string, int> LoadUserConfig(const std::string& userConfigFilePath);

/**
 * Method to create LlmConfig Instance from model-configuration and user-configuration dictionaries
 * @param config Model Configuration file which contains model related details like llmPrefix,
 * modelTag etc.
 * @return A LlmConfig file which can be used to construct an LLM instance.
 */
LlmConfig GetConfig(std::unordered_map<std::string, std::string> config,
                    std::unordered_map<std::string, int> userConfig);

/**
 * Method to scan a rolling buffer for any of the configured stop-words.
 * @param buffer    The concatenated recent tokens.
 * @param stopWords The list of stop-words.
 * @param outWord   (out) the first stop-word found.
 * @return          true if any stop-word appears in buffer.
 */
bool ContainsStopWord(const std::string& buffer,
                      const std::vector<std::string>& stopWords,
                      std::string& outWord);
} /* namespace Llm::Test::Utils */

#endif /* LLM_TEST_UTILS_HPP */
