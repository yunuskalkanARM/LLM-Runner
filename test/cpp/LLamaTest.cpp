//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
//
#define CATCH_CONFIG_MAIN

#include <catch.hpp>
#include "LLM.hpp"
#include "LlamaImpl.hpp"
#include <list>
#include <fstream>
#include <sstream>
#include <unordered_map>

// Function to parse the configuration file
std::unordered_map <std::string, std::string> LoadConfig(const std::string &configFilePath) {
    std::unordered_map <std::string, std::string> config;
    std::ifstream file(configFilePath);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open config file: " + configFilePath);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue; // Skip empty lines and comments

        size_t delimiterPos = line.find('=');
        if (delimiterPos != std::string::npos) {
            std::string key = line.substr(0, delimiterPos);
            std::string value = line.substr(delimiterPos + 1);
            config[key] = value;
        }
    }

    return config;
}

//ToDo AFAIC Several variables could also be read-in at compile time e.g. embeddings, nLen etc.
// Variables that need to be set after config file parsing
static constexpr int embeddings = 150;
static constexpr int tokens = 0;
static constexpr int sequenceMax = 1;
std::string testModelsDir = TEST_MODELS_DIR;
std::string modelPath =
        testModelsDir + "/model.gguf";
std::list <std::string> STOP_WORDS;
std::string llmPrefix;
std::string modelTag;
int nLen = 1024;

// Function to load configuration before tests
void InitializeConfig() {
    std::string configFilePath = CONFIG_FILE_PATH;
    std::unordered_map <std::string, std::string> config = LoadConfig(configFilePath);

    llmPrefix = config["llmPrefixDefault"];
    modelTag = config["modelTagDefault"];

    // Parse stopWordsDefault into a list
    std::istringstream stopWordsStream(config["stopWordsDefault"]);
    std::string word;
    while (std::getline(stopWordsStream, word, ',')) {
        STOP_WORDS.push_back(word);
    }
}

// Call InitializeConfig before tests run
struct ConfigInitializer {
    ConfigInitializer() {
        InitializeConfig();
    }
} configInitializer; // Global instance ensures initialization before tests

/**
 * Simple query->response test
 * ToDo Replace with more sophisticated context tests if/when reset context is available in Cpp layer
 */
TEST_CASE("Test Query Response") {
    std::string response;
    int nCur = 0;
    const std::string question = "What is the capital of France?" + modelTag;
    const std::string prefixedQuestion = llmPrefix + question;

    LLM<LlamaImpl> llm;
    auto *model = llm.LoadModel<llama_model>(modelPath.c_str());
    llm.BackendInit();

    auto *context = llm.NewContext<llama_context, llama_model>(model, 2);
    auto batch = llm.NewBatch<llama_batch>(embeddings, tokens, sequenceMax);
    nCur = llm.CompletionInit<llama_context, llama_batch>(prefixedQuestion, context, &batch, 0);

    while (nCur <= nLen) {
        std::string s = llm.CompletionLoop<llama_context, llama_batch>(context, &batch, nCur, nLen);
        if ((std::find(STOP_WORDS.begin(), STOP_WORDS.end(), s) != STOP_WORDS.end())) {
            break;
        }
        response += s;
    }
    CHECK(response.find("Paris") != std::string::npos);
}

/**
 * Test Load Empty Model returns nullptr
 */
TEST_CASE("Test Load Empty Model") {
    std::string emptyModelPath;
    LLM<LlamaImpl> llm;
    auto *model = llm.LoadModel<llama_model>(emptyModelPath.c_str());
    CHECK(model == nullptr);
}
