//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#include "LlmImpl.hpp"
#include "LlmUtils.hpp"

#include <sstream>

// Function to create the configuration file from CONFIG_FILE_PATH
std::vector<std::string> GetTestConfigStopWords(LlmConfig& configTest)
{
    std::string configFilePath     = CONFIG_FILE_PATH;
    std::string userConfigFilePath = USER_CONFIG_FILE_PATH;
    auto config                    = Llm::Test::Utils::LoadConfig(configFilePath);
    std::stringstream stopWordsStream;
    stopWordsStream << config["stopWords"];
    std::string word;
    std::vector<std::string> STOP_WORDS;
    while (std::getline(stopWordsStream, word, ',')) {
        STOP_WORDS.push_back(word);
    }
    std::string testModelsDir = TEST_MODELS_DIR;
    std::string modelPath     = testModelsDir + "/" + config["llmModelName"];
    config["modelPath"]       = modelPath;
    auto userConfig           = Llm::Test::Utils::LoadUserConfig(userConfigFilePath);
    configTest               = Llm::Test::Utils::GetConfig(config, userConfig);
    configTest.SetModelPath(modelPath);
    return STOP_WORDS;
}
/**
 * Simple query->response test
 * ToDo Replace with more sophisticated context tests if/when reset context is available in Cpp
 * layer
 */
TEST_CASE("Test Llm-Wrapper class")
{
    LlmConfig configTest{};
    auto stopWords = GetTestConfigStopWords(configTest);

    std::string response;
    std::string question         = configTest.GetUserTag() +"What is the capital of France?" +
                                   configTest.GetEndTag() + configTest.GetModelTag();
    std::string prefixedQuestion = configTest.GetLlmPrefix() + question;
    LLM llm;

    SECTION("Simple Query Response")
    {
        llm.LlmInit(configTest);
        llm.Encode(prefixedQuestion);
        bool stop = false;
        REQUIRE(!stopWords.empty());
        size_t t_maxWordLength = std::max_element(
            stopWords.begin(), stopWords.end(),
            [](auto const& a, auto const& b){
                return a.size() < b.size();
            }
        )->size();
        std::string t_tokenCache, t_breakWord;

        while (llm.GetChatProgress() < 100) {
            std::string s = llm.NextToken();
            t_tokenCache += s;
            response += s;
            if(Llm::Test::Utils::ContainsStopWord(t_tokenCache, stopWords, t_breakWord)) {
                size_t pos = response.find(t_breakWord);
                response.erase(pos);
                break;
            }
            if(t_tokenCache.size() > t_maxWordLength) {
                t_tokenCache.erase(0, t_tokenCache.size() - t_maxWordLength);
            }
        }
        CHECK(response.find("Paris") != std::string::npos);
    }

    /**
     * Test Load Empty Model returns nullptr
     */
    SECTION("Test Load Empty Model")
    {
        std::string emptyString;
        configTest.SetModelPath(emptyString);
        REQUIRE_THROWS(llm.LlmInit(configTest));
    }

    llm.FreeLlm();
}
