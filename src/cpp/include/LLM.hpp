//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLM_LLM_HPP
#define LLM_LLM_HPP

/**
 * Interface to LLM
 * Contains generic LLM functions
 * Should invoke whatever implementation this is initialised with
 */

template<typename T>
class LLM
{
private:

public:

    /**
     * Stores partial or accumulated token characters between inference steps
     */
    std::string cached_token_chars;

    /**
     * Function to load the chosen LLM model to memory
     * @tparam P model type
     * @param path_to_model path to the model location
     * @return LLM model pointer
     */
    template<typename P>
    P *LoadModel(const char *path_to_model)
    {
        return ((T *) this)->template LoadModel<P>(path_to_model);
    }

    /**
     * Frees the memory holding the LLM model
     * @tparam P model type
     * @param model LLM model pointer
     */
    template<typename P>
    void FreeModel(P *model)
    {
        ((T *) this)->template FreeModel<P>(model);
    }

    /**
     * Function to create a new LLM context object in memory
     * @tparam P LLM context type
     * @tparam V LLM model type
     * @param model LLM model instance
     * @param numThreads number of threads
     * @return  context pointer
     */
    template<typename P, typename V>
    P *NewContext(V *model, int numThreads)
    {
        return ((T *) this)->template NewContext<P, V>(model, numThreads);
    }

    /**
     * Free up the memory that is storing the LLM context
     * @tparam P LLM context type
     * @param context LLM context pointer
     */
    template<typename P>
    void FreeContext(P *context)
    {
        ((T *) this)->template FreeModel<P>(context);
    }

    /**
    * Function to initialize the LLM backend
    */
    void BackendInit()
    {
        ((T *) this)->BackendInit();
    }

    /**
     * Function to free up the memory storing the backend
     */
    void BackendFree()
    {
        ((T *) this)->BackendFree();
    }

    /**
     * Function to free up the memory storing the batch instance
     * @tparam P LLM batch type
     * @param batch LLM Batch object pointer
     */
    template<typename P>
    void FreeBatch(P &batch)
    {
        ((T *) this)->template FreeBatch<P>(batch);
    }

    /**
     * Function to retrieve the LLM encode timings
     * @tparam P LLM context type
     * @param context LLM Context object pointer
     * @return encode timings
     */
    template<typename P>
    float GetEncodeTimings(P *context)
    {
        return ((T *) this)->template GetEncodeTimings<P>(context);
    }

    /**
     * Function to retrieve the LLM decode timings
     * @tparam P LLM context type
     * @param context LLM Context object pointer
     * @return decode timings
     */
    template<typename P>
    float GetDecodeTimings(P *context)
    {
        return ((T *) this)->template GetDecodeTimings<P>(context);
    }

    /**
     * Function to reset the LLM timings
     * @tparam P LLM context type
     * @param context LLM context object pointer
     */
    template<typename P>
    void ResetTimings(P *context)
    {
        ((T *) this)->template ResetTimings<P>(context);
    }

    /**
    * Function to print the system info
    * @return system information
    */
    const char *SystemInfo()
    {
        return ((T *) this)->SystemInfo();
    }

    /**
     * Function to perform KV Cache clear
     * @tparam P LLM context type
     * @param context LLM Context object pointer
     */
    template<typename P>
    void KVCacheClear(P *context)
    {
        ((T *) this)->template KVCacheClear<P>(context);
    }

    /**
     * Function to removes all tokens that belong to the specified sequence and have positions in [p0, p1)
     * @tparam P LLM context object type
     * @param context LLM Context object
     * @param p0 starting token index (inclusive)
     * @param p1 ending token index (exclusive)
     */
    template<typename P>
    void KVCacheSeqRm(P *context, int p0, int p1)
    {
        ((T *) this)->template KVCacheSeqRm<P>(context, p0, p1);
    }

    /**
     * Function to tokenize the initial prompt
     * @tparam P LLM model type
     * @tparam V LLM tokens container
     * @param model pointer to the LLM model
     * @param text prompt text to be tokenized
     * @param textLength length of the prompt text in bytes (or characters)
     * @param tokens pointer to the container/array that will hold the resulting tokens
     * @param maxNumTokens maximum number of tokens
     * @param addSpecial if `true`, includes special tokens (e.g., BOS/EOS) in the output
     * @param parseSpecial if `true`, parses special tokens directly from the prompt text
     * @return length of original prompt
     */
    template<typename P, typename V>
    int GetInitialPromptLength(P *model, const char *text, int32_t textLength, V *tokens,
                               int32_t maxNumTokens, bool addSpecial, bool parseSpecial)
    {
        return ((T *) this)->template GetInitialPromptLength<P, V>(model, text, textLength,
                                                                   tokens, maxNumTokens, addSpecial, parseSpecial);
    }

    /**
     * Function to create a new batch object
     * @tparam P LLM batch type
     * @param embeddings embedding dimension for each token
     * @param numTokens maximum number of tokens in the batch
     * @param numSequenceMax maximum number of sequences or contexts
     * @return newly created batch object
     */
    template<typename P>
    P NewBatch(int embeddings, int numTokens, int numSequenceMax)
    {
        return ((T *) this)->template NewBatch<P>(embeddings, numTokens, numSequenceMax);
    }

    /**
     * Function used for encoding the prompt text
     * @tparam P LLM Context Type
     * @tparam V LLM Batch Type
     * @param text input prompt text to be encoded
     * @param context pointer to the LLM context
     * @param batch pointer to the LLM batch
     * @param startPos start position of the text that will be used
     * @return number of tokens if successful otherwise error code
     */
    template<typename P, typename V>
    int CompletionInit(std::string text, P *context, V *batch, int startPos)
    {
        return ((T *) this)->template CompletionInit<P, V>(text, context, batch, startPos);
    }

    /**
     * Main inference loop, returns each token of response from llm
     * @tparam P LLM context type
     * @tparam V LLM batch type
     * @param context LLM context object pointer
     * @param batch LLM batch object pointer
     * @param nCur reference to the current token index in the sequence
     * @param nLen reference to the total number of tokens to generate
     * @return newly generated token as a string
     */
    template<typename P, typename V>
    std::string CompletionLoop(P *context, V *batch, int &nCur, int &nLen)
    {
        return ((T *) this)->template CompletionLoop<P, V>(context, batch, nCur, nLen);
    }
};

#endif //LLM_LLM_HPP
