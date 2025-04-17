//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

package com.arm.llm;

import java.util.List;
import java.util.concurrent.Flow;
import java.util.concurrent.SubmissionPublisher;
import java.util.concurrent.atomic.AtomicBoolean;

public class Llama extends SubmissionPublisher<String>
{
    static
    {
        try
        {
            System.loadLibrary("arm-llm-jni");
        } catch (UnsatisfiedLinkError e)
        {
            System.err.println("Llama: Failed to load library: arm-llm-jni");
            e.printStackTrace();
        }
    }

    private long llmContext = 0;
    private long modelPointer = 0;
    private String modelTag = "";
    private static final int embeddings = 0;
    private static final int tokens = 150;
    private static final int sequenceMax = 1;
    private static final int nLen = 1024;
    private int nCur = 0;
    private long batch = 0;
    private List<String> stopWords = null;
    private String cachedToken = "";
    private String emitToken = "";
    private String llmPrefix = "";
    private int numThreads;

    //ToDo Create a session manager to manage a conversion instead of "evaluatedOnce"
    private AtomicBoolean evaluatedOnce = new AtomicBoolean(false);

    // Native method declarations

    /**
     * Method for loading LLM model
     *
     * @param pathToModel file path for loading model
     * @return pointer to loaded model
     */
    public native long loadModel(String pathToModel);

    /**
     * Method for freeing LLM model
     *
     * @param modelPtr to free model
     * @param contextPtr for freeing up LLM context
     */
    public native void freeModel(long modelPtr,long contextPtr);

    /**
     * Method for getting encode timing
     *
     * @param contextPtr LLM context pointer to the loaded context
     * @return timings in tokens/s for encoding prompt
     */
    public native float getEncodeTimings(long contextPtr);

    /**
     * Method for getting decode timing
     *
     * @param contextPtr LLM context pointer to the loaded context
     * @return timings in tokens/s for decoding prompt
     */
    public native float getDecodeTimings(long contextPtr);

    /**
     * Method for getting a new llama context
     *
     * @param modelPtr to loaded model
     * @param numThreads number of threads to use
     * @return pointer to LLM context loaded
     */
    public native long newContext(long modelPtr, int numThreads);

    /**
     * Method for clearing previous chat history from llama
     *
     * @param context LLM context pointer to the loaded context
     */
    private native void kvCacheClear(long context);

    /**
     * Method for clearing previous chat history until specified point from llama
     *
     * @param context LLM context pointer to the loaded context
     * @param startPos starting index from which to delete context memory (inclusive)
     * @param lastPos  the index upto which to clear (exclusive)
     */
    private native void kvCacheSeqRm(long context, int startPos, int lastPos);

    /**
     * Method for resetting timing information
     *
     * @param contextPtr LLM context pointer to the loaded context
     */
    public native void resetTimings(long contextPtr);

    /**
     * Method for getting a new sampler
     *
     * @param contextPtr LLM context pointer to the loaded context
     * @return pointer to sampler
     */
    public native long newSampler(long contextPtr);

    /**
     * Method to get llmPrefix length in terms of tokens
     *
     * @param modelPtr   pointer to LLM model
     * @param textLength length of initial prompt
     * @param text       LLM Prefix to be encoded
     * @param addSpecial bool for optional special character at end of prompt
     * @return length of original prompt
     */
    public native int getInitialPromptLength(long modelPtr, int textLength, String text, boolean addSpecial);

    /**
     * Method to get initializes the llama backend
     */
    public native void backendInit();

    /**
     * Method to create new batch
     *
     * @param numTokens number of allowed tokens in the batch
     * @param embeddings number of allowed embeddings in the batch
     * @param numSequenceMax number of sequences allowed
     * @return newly created batch object
     */
    public native long newBatch(int numTokens, int embeddings, int numSequenceMax);


    /**
     * Method to Encode the given text and return number of tokens in prompt
     *
     * @param text     the prompt to be encoded
     * @param context  pointer to llama_context instance
     * @param batch    pointer to llama batch of type llama_batch
     * @param startPos starting index of positional embeddings from which to populate current question
     * @return number of tokens if successful otherwise error code
     */
    public native int completionInit(
            String text,
            long context,
            long batch,
            int startPos
    );

    /**
     * Method to decode answers one by one, once prefill stage is completed
     *
     * @param context    pointer to llama_context instance
     * @param batch      pointer to llama batch of type llama_batch
     * @param nLen       max length of context memory to be filled
     * @param currentPos starting index of positional embeddings to populate current decoded token
     * @return generated token as a string
     */
    public native String completionLoop(
            long context,
            long batch,
            int nLen,
            int currentPos
    );

    /**
     *Method to separate Initialization from constructor
     *
     *@param llamaConfig type configuration file to load model
     */
    public void llmInit(LlamaConfig llamaConfig)
    {
        this.modelPointer = loadModel(llamaConfig.getModelPath());
        this.numThreads = llamaConfig.getNumThreads();
        this.llmContext = newContext(modelPointer, numThreads);
        this.batch = newBatch(tokens, embeddings, sequenceMax);
        this.stopWords = llamaConfig.getStopWords();
        this.modelTag = llamaConfig.getModelTag();
        this.llmPrefix = llamaConfig.getLlmPrefix();
    }

    /**
     *Method to assing a new subscriber to this publisher
     *
     *@param subscriber subscriber that will receive published tokens
     */
    public void setSubscriber(Flow.Subscriber<String> subscriber)
    {
        System.out.println("subscribed set from llama");
        this.subscribe(subscriber);
    }

    /**
     * Method to get response of a query asynchronously
     *
     * @param Query the prompt asked
     */
    public  void  sendAsync(String Query)
    {

        String query = "";
        AtomicBoolean stop = new AtomicBoolean(false);
        if (evaluatedOnce.get())
            query = Query + modelTag;
        else
            query = llmPrefix + Query + modelTag;
        nCur += completionInit(query, this.llmContext, this.batch, nCur);
        evaluatedOnce.set(true);
        while (nCur <= nLen)
        {
            String s = completionLoop(this.llmContext, this.batch, nCur, nLen);
            stop.set(inspectWord(s));
            if (stop.get())
            {
                emitToken = "<eos>";
                ++nCur;
                this.submit(emitToken);

                break;
            }
            ++nCur;
            this.submit(emitToken);
        }
    }

    /**
     * Method to find any stop-Words or partial stop-Word present in current token
     *
     * @param str current token decoded
     * @return boolean for detection of stop word
     */
    private boolean inspectWord(String str)
    {
        boolean stopWordTriggered = false;
        String evaluationString = this.cachedToken + str;
        // if stopWord is in evaluationString break loop
        for (String word : stopWords)
        {
            if (evaluationString.contains(word))
            {
                stopWordTriggered = true;
                emitToken = "";
                cachedToken = "";
                return stopWordTriggered;
            }
        }
        emitToken = evaluationString;
        for (String word : stopWords)
        {
            String partialWord = word;
            partialWord = partialWord.substring(0, partialWord.length() - 1);
            while (!partialWord.isEmpty())
            {
                if (evaluationString.endsWith(partialWord))  // if the beginning for stop word coincides with end of emitted token dont emit current token
                {
                    emitToken = "";
                    break;
                } else
                {
                    partialWord = partialWord.substring(0, partialWord.length() - 1);
                }
            }
        }
        this.cachedToken = emitToken.isEmpty() ? evaluationString : "";
        return stopWordTriggered;
    }

    /**
     * Method to reset conversation history
     */
    public void resetContext()
    {

        int nPrefix = getInitialPromptLength(this.modelPointer, this.llmPrefix.length(), this.llmPrefix, true);
        if (nPrefix < 0)
        {
            nPrefix = 0;
        }
        kvCacheSeqRm(this.llmContext, nPrefix, -1);
        resetTimings(this.llmContext);
        nCur = nPrefix;
    }

    /**
     * Method to get response of a query synchronously
     *
     * @param Query the prompt asked
     * @return response of LLM
     */
    public String send(String Query)
    {
        String response = "";
        String query = "";
        boolean stop = false;
        if (evaluatedOnce.get())
            query = Query + modelTag;
        else
            query = llmPrefix + Query + modelTag;
        nCur += completionInit(query, this.llmContext, this.batch, nCur);
        evaluatedOnce.set(true);
        while (nCur <= nLen)
        {

            String s = completionLoop(this.llmContext, this.batch, nCur, nLen);
            stop = inspectWord(s);
            if (!stop)
            {
                response += emitToken;
            } else
            {
                ++nCur;
                break;
            }
            ++nCur;
        }

        return response;
    }

    /**
     * Method to get current encode timings
     *
     * @return encode timings in tokens/s
     */
    public float getEncodeRate()
    {
        return getEncodeTimings(this.llmContext);
    }

    /**
     * Method to get current decode timings
     *
     * @return decode timings in tokens/s
     */
    public float getDecodeRate()
    {
        return getDecodeTimings(this.llmContext);
    }

    /**
     * Sets the LLM prefix used for query processing
     */
    public void setLlmPrefix(String llmPrefix)
    {
        this.llmPrefix = llmPrefix;
    }

    /**
     * Sets the LLM ModelTag
     */
    public void setLlmModelTag(String newTag)
    {
        this.modelTag = newTag;
    }
    /**
     * Method to free model from memory
     */
    public void freeModel()
    {
        evaluatedOnce.set(false);
        freeModel(this.modelPointer,this.llmContext);
        this.close();  // Publisher is closed
    }


}

