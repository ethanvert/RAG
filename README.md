# RAG Demo App: Presidential History Chatbot

## What is this?

Welcome to the RAG Demo App, a chatbot that's like having a White House insider in your pocket (minus the security clearance)! This project showcases the power of Retrieval-Augmented Generation (RAG) in creating a specialized AI assistant focused on U.S. Presidential history, governmental structure, and up-to-date political insights.

## What does it do?

Our chatbot is designed to:

- Educate users on U.S. Presidential history
- Provide accurate information about governmental structure and foundations
- Reference recent resources that might not be in a pre-trained model's dataset

For example, you can ask questions like:
- "What did Teddy Roosevelt say about nature?"
- "What is Joe Biden doing about Ukraine now?"

## How does it work?

1. **User Query**: The app takes in a user's question about U.S. politics or history.

2. **Smart Search**: It then searches through a curated collection of:
   - Speeches delivered by U.S. Presidents
   - Biographies of Presidents and First Ladies
   - Q&A from the White House website

3. **Context Building**: The most relevant hits (including context and metadata) are formatted and fed into an LLM.

4. **AI Magic**: The LLM, armed with the original query, relevant context, and specific instructions, crafts a response.

## Technologies Used

- **LangChain**: For orchestrating the whole show
- **HuggingFace**: Providing models (Embedding, Tokenizer, Foundation Model)
- **LLM**: LLaMa 3.2 3B for Q&A (Note: In production, I'd use a model with >8B parameters)
- **ChromaDB**: For vector store and retrieval

## The Process: From Concept to Chatbot

### 1. Data Preprocessing
- Collected and curated a diverse set of presidential speeches, biographies, and official Q&As
- Processed and formatted the data for optimal retrieval

### 2. RAG Architecture Implementation
- Designed a system that efficiently retrieves relevant information based on user queries
- Integrated the retrieval process with the LLM for coherent and informed responses

### 3. Continuous Improvement
- Regularly evaluated and fine-tuned the system for better performance
- Focused on enhancing retrieval accuracy and response quality

## Challenges and Lessons Learned

### Challenges:
- Developing a robust retriever that goes beyond surface-level content searching
- Creating an effective evaluation process for both retrieval and response quality
- Optimizing the system to work within the constraints of Google Colab for extended coding sessions

### Key Takeaways:
1. **Retrieval is King**: A robust retriever is crucial for high-quality answers. Focus on retrieving the most important information (80/20 rule).
2. **Evaluation Matters**: Invest time in creating ground truth retrieval examples and hand-made Q&A pairs for thorough testing.
3. **LLM Considerations**: 
   - LLMs can be resource-intensive. Consider quantization techniques to reduce computational costs.
   - For handling long messages, state-of-the-art LLMs (like those from OpenAI, Cohere, or Anthropic) might be necessary.
4. **It's Still Just a Web App**: At its core, this AI system is similar to other web applications in terms of resource management and API design.

## Results and Comparisons

While our specialized RAG Demo App shines in its focused domain, I found that ChatGPT-4 outperforms our bot in several areas:
- General historical Q&A
- Comparative analysis
- Complex reasoning tasks

However, our system excels in:
- Providing specialized information on U.S. Presidential history
- Offering up-to-date insights on current governmental affairs
- Delivering consistent and focused responses in its specialized domain

## Future Improvements

1. Implement more advanced retrieval techniques (e.g., Langchain SelfQueryRetriever, LlamaIndex AutoRetriever)
2. Expand the knowledge base to cover more aspects of U.S. politics and history
3. Explore fine-tuning options for even more accurate and contextually relevant responses
4. Optimize deployment for better scalability and reduced latency
5. Expand front end from basic Gradio demo interface

## Get Involved

I'm always looking to improve! If you're interested in contributing or have suggestions, please:
1. Fork the repository
2. Create a new branch for your feature
3. Submit a pull request with a clear description of your changes

## Contact

For any questions or feedback, reach out  at <a href="mailto:ethan.vertal@gmail.com">ethan.vertal@gmail.com</a>.

Demo Coming Soon!
