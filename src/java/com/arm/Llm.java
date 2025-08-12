//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

package com.arm;

import java.util.List;
import java.util.concurrent.Flow;
import java.util.concurrent.SubmissionPublisher;
import java.util.concurrent.atomic.AtomicBoolean;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.IOException;
/**
 * Llm class that extends the SubmissionPublisher
 */
public class Llm extends SubmissionPublisher<String>
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

    /**
     * Method to create Llm cpp instance from params.
     * @param ConfigPathStr path to config.json file as a string
     * @return null
     */
    public native void llmInit(String ConfigPathStr);

    /**
     * Method for freeing LLM model
     * @param modelPtr to free model
     */
    private native void freeLlm();

    /**
     * Public method for getting encode timing
     * @return timings in tokens/s for encoding prompt
     */
    public native float getEncodeRate();

    /**
     * Public method for getting decode timing
     * @return timings in tokens/s for decoding prompt
     */
    public native float getDecodeRate();

    /**
     * Private method for resetting conversation history
     */
    public native void resetContext();

    /**
     * Method for resetting timing information
     */
    public native void resetTimings();

    /**
     * Method to encode the given text
     * @param text     the prompt to be encoded
     */
    private native void encode(String text);

    /**
     * Method to get Next Token once encoding is done.
     * This Method needs to be called in a loop while monitoring for Stop-Words.
     * @return next Token as String
     */
    private native String getNextToken();

    /**
     * Method to get chat Progress in percentage
     * @return chat progress as an integer percentage
     */
    public native int getChatProgress();

    /**
     * Method to decode answers one by one, once prefill stage is completed
     * @param nPrompts     prompt length used for benchmarking
     * @param nEvalPrompts number of generated tokens for benchmarking
     * @param nMaxSeq      sequence number
     * @param nRep         number of repetitions
     * @return string containing results of the benchModel
     */
    public native String benchModel(
            int nPrompts,
            int nEvalPrompts,
            int nMaxSeq,
            int nRep
    );

    /**
     * Method to get framework type
     * @return string framework type
     */
    public native String getFrameworkType();

    /**
     * Method to set subscriber
     * @param subscriber set from llama
     */
    public void setSubscriber(Flow.Subscriber<String> subscriber)
    {
        this.subscribe(subscriber);
    }

    /**
     * Method to get response of a query asynchronously
     * @param Query the prompt asked
     */
    public void  sendAsync(String Query)
    {
        encode(Query);
        while (getChatProgress()<100)
        {
            String s = getNextToken();
            if (s.equals("<eos>"))
            {
                // needed for showing end of stream, Closing publisher will result in error
                // for next query
                this.submit(s);
                break;
            }
            this.submit(s);
        }
    }

    /**
     * Method to get response of a query synchronously
     * @param Query the prompt asked
     * @return response of LLM
     */
    public String send(String Query)
    {
        String response = "";
        encode(Query);
        while (getChatProgress()<100)
        {
            String s = getNextToken();
            response += s;
            if (s.equals("<eos>"))
              break;
        }
        return response;
    }

    /**
    * Method to free model from memory
    */
    public void freeModel()
    {
        freeLlm();
        this.close();
    }
}
