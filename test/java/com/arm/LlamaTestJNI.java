//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

package com.arm;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertFalse;
import static org.junit.Assume.assumeTrue;

import org.junit.Test;
import org.junit.BeforeClass;

import com.arm.llm.Llama;
import com.arm.llm.LlamaConfig;

import java.io.*;
import java.util.*;

public class LlamaTestJNI {
    private static final String modelDir = System.getProperty("model_dir");
    private static final String configFilePath = System.getProperty("config_file");
    private static final Map<String, String> variables = new HashMap<>();
    private static final String LLAMA_MODEL_NAME = "model.gguf";
    private static final int numThreads = 4;

    private static String modelTag = "";
    private static String modelPath = "";
    private static String llmPrefix = "";
    private static List<String> stopWords = new ArrayList<String>();

    /**
     * Instead of matching the actual response to expected response,
     * check whether the response contains the salient parts of expected response.
     * Pass true to check match and false to assert absence of salient parts for negative tests.
     */
    private static void checkLlamaMatch(String response, String expectedResponse, boolean checkMatch) {
        boolean matches = response.contains(expectedResponse);
        if (!matches) {
            System.out.println("Response mismatch: response={" + response + "} expected={" + expectedResponse + "}");
        }
        if (checkMatch) {
            assertTrue(matches);
        } else {
            assertFalse(matches);
        }
    }

    /**
     * Loads variables from the specified configuration file.
     *
     * @param filePath Path to the configuration file.
     * @throws IOException If an I/O error occurs.
     */
    private static void loadVariables(String filePath) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (!line.contains("=")) continue;
                String[] parts = line.split("=", 2);
                if (parts[0].trim().equals("stopWordsDefault")) {
                    stopWords.clear(); // Ensure no duplicates on reloading
                    stopWords.addAll(Arrays.asList(parts[1].split(",")));
                } else {
                    variables.put(parts[0].trim(), parts[1].trim());
                }
            }
        } catch (FileNotFoundException e) {
            throw new IOException("Configuration file not found: " + filePath);
        } catch (Exception e) {
            throw new IOException("Error reading configuration file: " + e.getMessage());
        }
    }

    @BeforeClass
    public static void classSetup() throws IOException {
        if (modelDir == null) throw new RuntimeException("System property 'model_dir' is not set!");
        if (configFilePath == null)
            throw new RuntimeException("System property 'config_file' is not set!");

        loadVariables(configFilePath);
        modelTag = variables.get("modelTagDefault");
        llmPrefix = variables.get("llmPrefixDefault");
        modelPath = modelDir + "/" + LLAMA_MODEL_NAME;
    }

    @Test
    public void testConfigLoading() {
        LlamaConfig llamaConfig = new LlamaConfig(modelTag, stopWords, modelPath, llmPrefix, numThreads);
        assertTrue("Model tag is not empty", !llamaConfig.getModelTag().isEmpty());
        assertTrue("LLM prefix is not empty", !llamaConfig.getLlmPrefix().isEmpty());
        assertTrue("Stop words list is not empty", !llamaConfig.getStopWords().isEmpty());
    }

    @Test
    public void testLlmPrefixSetting() {
        LlamaConfig llamaConfig = new LlamaConfig(modelTag, stopWords, modelPath, llmPrefix, numThreads);
        Llama llama = new Llama();
        llama.llmInit(llamaConfig);

        String newModelTag = ("Ferdia");
        String newPrefix = "Transcript of a dialog, where the User interacts with an AI Assistant named " + newModelTag +
                ". " + newModelTag +
                " is helpful, polite, honest, good at writing and answers honestly with a maximum of two sentences. User:";

        llama.setLlmModelTag(newModelTag);
        llama.setLlmPrefix(newPrefix);

        String question = "What is your name?";
        String response = llama.send(question);
        checkLlamaMatch(response, "Ferdia", true);
        llama.freeModel();
    }

    @Test
    public void testInferenceWithContextReset() {
        LlamaConfig llamaConfig = new LlamaConfig(modelTag, stopWords, modelPath, llmPrefix, numThreads);
        Llama llama = new Llama();
        llama.llmInit(llamaConfig);

        String question1 = "What is the capital of Morocco?";
        String response1 = llama.send(question1);
        checkLlamaMatch(response1, "Rabat", true);

        // Resetting context should cause model to forget what country is being referred to
        llama.resetContext();

        String question2 = "What languages do they speak there?";
        String response2 = llama.send(question2);
        checkLlamaMatch(response2, "Arabic", false);

        llama.freeModel();
    }

    @Test
    public void testInferenceWithoutContextReset() {
        LlamaConfig llamaConfig = new LlamaConfig(modelTag, stopWords, modelPath, llmPrefix, numThreads);
        Llama llama = new Llama();
        llama.llmInit(llamaConfig);

        String question1 = "What is the capital of Morocco?";
        String response1 = llama.send(question1);
        checkLlamaMatch(response1, "Rabat", true);

        String question2 = "What languages do they speak there?";
        String response2 = llama.send(question2);
        checkLlamaMatch(response2, "Arabic", true);

        llama.freeModel();
    }

    @Test
    public void testInferenceHandlesEmptyQuestion() {
        LlamaConfig llamaConfig = new LlamaConfig(modelTag, stopWords, modelPath, llmPrefix, numThreads);
        Llama llama = new Llama();
        llama.llmInit(llamaConfig);

        String question1 = "What is the capital of Morocco?";
        String response1 = llama.send(question1);
        checkLlamaMatch(response1, "Rabat", true);

        // Send an empty prompt to simulate blank recordings or non-speech tokens being returned by speech recognition;
        // then ask follow-up questions to ensure previous context persists when an empty prompt is injected in the conversation.
        String emptyResponse = llama.send(""); // ToDo may revisit this to add an expected answer

        String question2 = "What languages do they speak there?";
        String response2 = llama.send(question2);
        checkLlamaMatch(response2, "Arabic", true);
        checkLlamaMatch(response2, "French", true);

        llama.freeModel();
    }

    @Test
    public void testMangoSubtractionLongConversation() {

        LlamaConfig llamaConfig = new LlamaConfig(modelTag, stopWords, modelPath, llmPrefix, numThreads);
        Llama llama = new Llama();
        llama.llmInit(llamaConfig);

        // 35 was determined to be upper limit for storing context but to avoid excessively long test runtime we cap at 20
        int originalMangoes = 5;
        int mangoes = originalMangoes;

        // Set the initial ground truth in the conversation.
        String initialContext = "There are " + originalMangoes + " mangoes.";
        String initResponse = llama.send(initialContext);
        String originalQuery = "How many mangoes were there originally?";
        String subtractQuery = "Subtract 1 mango.";

        // **Assert that the model acknowledges the initial count of mangoes.**
        checkLlamaMatch(initResponse, String.valueOf(originalMangoes), true);

        // Loop to subtract 1 mango each iteration until reaching 0.
        for (int i = 1; i < originalMangoes; i++) {

            // Query to subtract one mango
            String subtractionResponse = llama.send(subtractQuery);
            mangoes -= 1;  // Update our expected count
            checkLlamaMatch(subtractionResponse, String.valueOf(mangoes), true);

            // Test if model still recalls the starting number
            if (i == originalMangoes - 1) {
                String response = llama.send(originalQuery);
                checkLlamaMatch(response, String.valueOf(originalMangoes), true);
                llama.resetContext();
            }

        }

        String postResetResponse = llama.send(originalQuery);
        checkLlamaMatch(postResetResponse, String.valueOf(originalMangoes), false);
        llama.freeModel();
    }

    @Test
    public void testInferenceRecoversAfterContextReset() {
        // Get model directory and config file path from system properties
        String modelDir = System.getProperty("model_dir");
        String configFilePath = System.getProperty("config_file");
        if (modelDir == null || configFilePath == null) {
            throw new RuntimeException("System properties for model_dir or config_file are not set!");
        }

        LlamaConfig llamaConfig = new LlamaConfig(modelTag, stopWords, modelPath, llmPrefix, numThreads);
        // Initialize Llama with the loaded config
        Llama llama = new Llama();
        llama.llmInit(llamaConfig);

        // First Question
        String question1 = "What is the capital of Morocco?";
        String response1 = llama.send(question1);
        checkLlamaMatch(response1, "Rabat", true);
        // Reset Context before second question
        llama.resetContext();

        // Second Question (After Reset)
        String question2 = "What languages do they speak there?";
        String response2 = llama.send(question2);
        checkLlamaMatch(response2, "Arabic", false);
        // Ask First Question Again. Note an additional reset is required to prevent the generic answer from previous question affecting new topic.
        llama.resetContext();
        String response3 = llama.send(question1);

        checkLlamaMatch(response3, "Rabat", true);
        String response4 = llama.send(question2);
        checkLlamaMatch(response4, "Arabic", true);
        checkLlamaMatch(response4, "French", true);

        // Free model after use
        llama.freeModel();
    }
}
