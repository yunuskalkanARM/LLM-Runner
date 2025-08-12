//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#include "LlmImpl.hpp"
#include <sstream>

// Function to create the configuration file from CONFIG_FILE_PATH
void SetupTestConfig(std::stringstream& stopWordsStream,
                     LlmConfig* configTest)
{
    std::string configFilePath     = CONFIG_FILE_PATH;
    std::ifstream configFile(configFilePath);
    std::stringstream buffer;
    buffer << configFile.rdbuf();  // Read file into stringstream
    std::string jsonContent = buffer.str();
    *configTest = LlmConfig(jsonContent);
    std::string testModelsDir = TEST_MODELS_DIR;
    std::string modelPath     = testModelsDir + "/" + configTest->GetModelPath();
    configTest->SetModelPath(modelPath);
}
/**
 * Simple query->response test
 * ToDo Replace with more sophisticated context tests if/when reset context is available in Cpp
 * layer
 */
TEST_CASE("Test Llm-Wrapper class")
{
    LlmConfig configTest{};
    std::stringstream stopWordsStream;
    SetupTestConfig(stopWordsStream, &configTest);

    std::string question         = "What is the capital of France?" ;
    LLM llm;

    SECTION("Simple Query Response")
    {
        std::string response;
        llm.LlmInit(configTest);
        llm.Encode(question);
        while (llm.GetChatProgress() < 100) {
            std::string s = llm.NextToken();
            if (s==llm.endToken)
                break;
            response += s;
        }
        std::cout<<"response is"<<response<<std::endl;
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
