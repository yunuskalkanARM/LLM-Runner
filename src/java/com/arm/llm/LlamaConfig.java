//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

package com.arm.llm;

import java.util.List;

public class LlamaConfig
{
    private static final String LLAMA_MODEL_NAME = "model.gguf";
    private String modelTag;
    private String modelPath;
    private String llmPrefix;
    private List<String> stopWords;
    private Integer numThreads;

    public LlamaConfig(String modelTag, List<String> stopWords, String modelPath, String llmPrefix, Integer numThreads)
    {
        this.modelTag = modelTag;
        this.stopWords = stopWords;
        this.modelPath = modelPath;
        this.llmPrefix = llmPrefix;
        this.numThreads = numThreads;
    }

    /**
     * Gets the number of threads
     *
     * @return The number of threads
     */
    public Integer getNumThreads()
    {
        return this.numThreads;
    }

    /**
     * Gets the model tag
     *
     * @return The model tag
     */
    public String getModelTag()
    {
        return this.modelTag;
    }

    /**
     * Gets the list of stop words
     *
     * @return The list of stop words
     */
    public List<String> getStopWords()
    {
        return this.stopWords;
    }

    /**
     * Gets the model path
     *
     * @return The model path
     */
    public String getModelPath()
    {
        return this.modelPath;
    }

    /**
     * Gets the LLM prefix
     *
     * @return The LLM prefix
     */
    public String getLlmPrefix()
    {
        return this.llmPrefix;
    }

    /**
     * Sets the number of threads
     *
     * @param numThreads The number of threads to set
     */
    public void setNumThreads(Integer numThreads)
    {
        this.numThreads = numThreads;
    }

    /**
     * Sets the model tag
     *
     * @param modelTag The model tag to set
     */
    public void setModelTag(String modelTag)
    {
        this.modelTag = modelTag;
    }

    /**
     * Sets the list of stop words
     *
     * @param stopWords The list of stop words to set
     */
    public void setStopWords(List<String> stopWords)
    {
        this.stopWords = stopWords;
    }

    /**
     * Sets the model path
     *
     * @param modelPath The model path to set
     */
    public void setModelPath(String modelPath)
    {
        this.modelPath = modelPath + "/" + LLAMA_MODEL_NAME;
    }

    /**
     * Sets the LLM prefix
     *
     * @param llmPrefix The LLM prefix to set
     */
    public void setLlmPrefix(String llmPrefix)
    {
        this.llmPrefix = llmPrefix;
    }
}

