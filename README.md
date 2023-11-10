# A-Guide-to-Retrieval-Augmented-LLM

Translated version

The emergence of ChatGPT has allowed us to see the capabilities of Large Language Model (LLM) in many aspects such as language and code understanding, human instruction following, and basic reasoning. However, the hallucination problem 
**[Hallucinations](https://machinelearningmastery.com/a-gentle-introduction-to-hallucinations-in-large-language-models/)** remains an important challenge facing current large language models. Simply put, the hallucination problem is when an LLM generates results that are incorrect, absurd, or inconsistent with reality. In addition, data freshness (Data Freshness) is another problem that occurs when LLM generates results, that is, LLM may not be able to give or give outdated answers to some time-sensitive questions. Retrieving external relevant information to enhance the generated results of LLM is a popular solution to solve the above problems. This solution is called Retrieval Augmented LLM (Retrieval Augmented LLM), sometimes also called Retrieval Augmented LLM. Retrieval Augmented Generation (RAG) for retrieval. This long article will give a relatively comprehensive introduction to the scheme of retrieval-enhanced LLM. The main contents include:

- The concept introduction, importance and problems solved by RA LLM
- The key modules of RA LLM and their implementation methods
- Some case studies and applications of RA LLM

This article can be regarded as a summary of my study in this field, so it may not be very professional and in-depth, and there will inevitably be some inaccuracies. Discussions are welcome.



# What is RA LLM

RA LLM (Retrieval Augmented LLM), simply put, is to provide an external database for LLM. For user questions (Query), through some IR(Information Retrieval) technology, first retrieve the user questions from the external database Relevant information, and then let LLM combine this relevant information to generate results. The figure below is a simple diagram of a retrieval-enhanced LLM.

![retrieval_augmented_llm_overall_architecture](assets/retrieval_augmented_llm_overall_architecture.png)

OpenAI research scientist Andrej Karpathy made a sharing about the current status of the GPT model at the Microsoft Build 2023 conference some time ago [State of GPT](https://www.youtube.com/watch?v=bZQun8Y4L2A&ab_channel=MicrosoftDeveloper). The first half of the speech shared how models such as ChatGPT are trained step by step. The second half mainly shared some application directions of the LLM model, including a brief introduction to the application direction of retrieval enhanced LLM. The picture below is the introduction to this direction shared by Andrej.

![retrieval_augmented_llm_in_state_of_gpt](assets/retrieval_augmented_llm_in_state_of_gpt.jpeg)

Traditional information retrieval tools, such as search engines like Google/Bing, only have retrieval capabilities (**Retrieval-only**). Now LLM embeds massive data and knowledge into its huge model parameters through the pre-training process, with Memory ability ( **Memory-only** ). From this perspective, retrieval-enhanced LLM is in the middle, combining LLM with traditional information retrieval, and loading relevant information into the working memory (**Working Memory**) of LLM through some information retrieval techniques, that is, the context window of LLM ( **Context Window** ), which is the maximum text input that LLM can accept in a single generation.

Not only did Andrej’s sharing mention the application method of enhancing LLM based on retrieval, but also from the research and summary of the technology stack of AI start-ups by some well-known investment institutions, we can also see the widespread application of retrieval-based LLM technology. For example, in June this year, Sequoia Capital published an article about the large language model technology stack [**The New Language Model Stack**](https://www.sequoiacap.com/article/llm-stack-perspective/) , which gave the results of a questionnaire survey of 33 AI start-ups it invested in. The survey results in the figure below show that about 88% of entrepreneurs said that they have used search-based enhanced LLM technology in their products. .

![sequoia_language_model_survey](assets/sequoia_language_model_survey.png)

Coincidentally, the famous American venture capital institution A16Z also published a summary article introducing the current LLM application architecture in June this year [**Emerging Architectures for LLM Applications**](https://a16z.com/emerging-architectures-for -llm-applications/), the following figure is the typical architecture of the current LLM application summarized in the article, among which the way **Contextual Data** at the top introduces LLM is an idea to enhance LLM through retrieval.

![emerging_llm_app_stack](assets/emerging_llm_app_stack.png)

# Retrieve problems solved by RA LLM

Why RA LLM with traditional information retrieval systems? In other words, what is the main problem solved by LLM based on retrieval enhancement?

## Long tail knowledge

Although the current LLM training data volume is already very large, often hundreds of GB of data and trillions of markers (Tokens). For example, the pre-training data of GPT-3 uses 300 billion markers, and LLaMA uses 1.4 trillion mark. The sources of training data are also very rich, such as Wikipedia, books, forums, codes, etc. The number of LLM model parameters is also very huge, ranging from billions, tens of billions to hundreds of billions. However, LLM can record in limited parameters. It is unrealistic to have all knowledge or information, and the coverage of training data is also limited. There will always be some long-tail knowledge that cannot be covered in the training data.
For some relatively general and popular knowledge, LLM can usually generate more accurate results, but for some long-tail knowledge, the responses generated by LLM are usually unreliable. This paper [Large Language Models Struggle to Learn Long-Tail Knowledge](https://arxiv.org/abs/2211.08411) at the ICML conference studies the accuracy of LLM for fact-based question answering and the accuracy of pre-training data. We found a strong correlation with the number of documents in related fields, that is, the greater the number of relevant documents in the pre-training data, the higher the accuracy of LLM's response to factual questions and answers. A simple conclusion can be drawn from this study - LLM's learning ability for long-tail knowledge is relatively weak. The picture below is the correlation curve drawn in the paper.

![llm_long_tail_evidence](assets/llm_long_tail_evidence.png)

In order to improve LLM's ability to learn long-tail knowledge, it is easy to think of adding more relevant long-tail knowledge to the training data, or increasing the number of parameters of the model. Although both methods do have certain effects, as mentioned above There is also experimental data support in the paper, but these two methods are uneconomical, that is, a large training data magnitude and model parameters are required to greatly improve the accuracy of LLM's response to long-tail knowledge. The retrieval method provides relevant information as context during LLM inference, which can not only achieve a better response accuracy, but also be a more economical way. The picture below shows the response accuracy of models of different sizes when relevant information is provided. Comparing the previous picture, we can see that for models of the same parameter magnitude, when a small number of relevant documents are provided to participate in pre-training, let The model utilizes relevant information during the inference phase, and its response accuracy is greatly improved.

![llm_long_tail_retrieval_method_performance](assets/llm_long_tail_retrieval_method_performance.png)

## private data

Most of the pre-training stages of general-purpose LLMs such as ChatGPT use public data and do not include private data, so some private domain knowledge is lacking. For example, if you ask ChatGPT about the internal knowledge of a certain enterprise, ChatGPT will most likely not know it or make it up randomly. Although private data can be added in the pre-training stage or used for fine-tuning, the training and iteration costs are high. In addition, research and practice have shown that LLM can leak training data through some specific attack methods. If the training data contains some private information, privacy information leakage is likely to occur. For example, the researchers of this paper [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805) extracted personal disclosures from the **GPT-2** model through the constructed Query. Name, email, phone number and address information, etc., even though this information may only appear once in the training data. The article also found that larger-scale models are more vulnerable to attacks than smaller-scale ones.

![llm_private_training_data_leak_example1](assets/llm_private_training_data_leak_example1.png)

If private data is used as an external database, LLM can directly retrieve relevant information from the external database when answering questions based on private data, and then combine the retrieved relevant information to answer. This eliminates the need for LLM to remember private knowledge in parameters through pre-training or fine-tuning, which not only saves training or fine-tuning costs, but also avoids the risk of private data leakage to a certain extent.

## Data freshness
Since the knowledge learned in LLM comes from training data, although the update cycle of most knowledge will not be very fast, there will still be some knowledge or information that is updated very frequently. The information LLM learns from pre-training data can easily become obsolete. For example, the GPT-4 model uses pre-training data as of 2021-09, so when it comes to events or information after this date, it will refuse to answer or give outdated or inaccurate responses. The following example asks GPT-4 who is the current CEO of Twitter. The reply given by GPT-4 is still Jack Dorsey, and it will remind itself that the reply may be out of date.

![ceo_of_twitter_gpt](assets/ceo_of_twitter_gpt.png)

If frequently updated knowledge is used as an external database for LLM to retrieve when necessary, the knowledge of LLM can be updated and expanded without retraining LLM, thus solving the problem of LLM data freshness.
## Source verification and interpretability
Often, the output generated by LLM does not give its source, making it difficult to explain why it is generated as it is. By providing external data sources to LLM and allowing it to generate based on the retrieved relevant information, a correlation is established between the generated results and the information source, so the generated results can be traced back to the reference source, making it interpretable and controllable. Sex is greatly enhanced. That is, you can know what relevant information LLM is based on to generate a reply. Bing Chat is a typical product that uses retrieval to enhance LLM output. The following figure shows a screenshot of the Bing Chat product. You can see that links to relevant information are given in the responses generated.

![bing_chat_screeshot](assets/bing_chat_screeshot.png)

An important step in using retrieval to enhance the output of LLM is to use some retrieval-related techniques to find relevant information fragments from external data, and then use the relevant information fragments as context for LLM to refer to when generating responses. One might say that as the context window ( **Context Window** ) of LLM becomes longer and longer, the step of retrieving relevant information is not necessary, and as much information as possible is provided directly in the context. For example, the maximum context length currently received by the GPT-4 model is 32K, and the maximum allowed context length of the Claude model is [100K](https://www.anthropic.com/index/100k-context-windows).
Although the context window of LLM is getting larger, the step of retrieving relevant information is still important and necessary. On the one hand, the current network architecture of LLM determines that the length of its context window has an upper limit and will not grow infinitely. In addition, the seemingly large context window actually holds relatively limited information. For example, the length of 32K may only be equivalent to the length of a university graduation thesis. On the other hand, studies have shown that providing a small amount of more relevant information leads to greater accuracy in LLM responses than providing a large amount of unfiltered information. For example, this paper [Lost in the Middle](https://arxiv.org/pdf/2307.03172.pdf) from Stanford University gives the following experimental results. You can see that the accuracy of the LLM reply increases with the context window. The number of documents provided increases and decreases.
![lost_in_the_middle_result](assets/lost_in_the_middle_result.png)
Retrieval technology is used to find the information fragments most relevant to the input question from a large amount of external data. While providing a reference for LLM to generate responses, it also filters out the interference of some irrelevant information to a certain extent to improve the accuracy of generated responses. Furthermore, the larger the context window, the higher the inference cost. Therefore, the introduction of relevant information retrieval steps can also reduce unnecessary reasoning costs.

# Key modules
In order to build a retrieval-enhanced LLM system, the key modules that need to be implemented and the problems that need to be solved include:
- **Data and Index Module**: How to handle external data and build indexes
- **Query and retrieval module**: How to retrieve relevant information accurately and efficiently
- **Response Generation Module**: How to utilize retrieved relevant information to enhance the output of LLM
## Data and index module
### data collection
The role of the data acquisition module is generally to convert external data from multiple sources, types and formats into a unified document object (Document Object) to facilitate processing and use in subsequent processes. In addition to containing the original text content, the document object generally also carries the metainformation of the document (Metadata), which can be used for later retrieval and filtering. Meta information includes but is not limited to:
- Time information, such as document creation and modification time
- Title, keywords, entities (people, places, etc.), text categories and other information
- Text summaries and abstracts
Some meta-information can be obtained directly, and some can rely on NLP technology, such as keyword extraction, entity recognition, text classification, text summarization, etc. Either traditional NLP models and frameworks can be used, or they can be implemented based on LLM.

![data_ingestion_overview](assets/data_ingestion_overview.png)

The sources of external data may be diverse, such as
- Various Doc documents, Sheets, Slides presentations, Calendar schedules, Drive files, etc. in Google suite
- Data from chat communities such as Slack and Discord
- Code files hosted on Github, Gitlab
- Various documents on Confluence
- Data from Web pages
- Data returned by the API
- local files

The types and file formats of external data may also be diverse, such as
- From the perspective of data type, including plain text, tables, presentation documents, codes, etc.
- From the perspective of file storage format, including txt, csv, pdf, markdown, json and other formats
External data may be multilingual, such as Chinese, English, German, Japanese, etc. In addition, it may also be multi-modal. In addition to the text modality discussed above, it also includes pictures, audio, video and other modalities. However, the external data discussed in this article will be limited to text modality.
When building a data acquisition module, data from different sources, types, formats, and languages ​​may require different reading methods.

### Text chunking
Text chunking is the process of cutting long text into smaller pieces, such as cutting a long article into relatively short paragraphs. So why text chunking? On the one hand, the current context length of LLM is limited. Directly putting all of a long article as relevant information into the context window of LLM may exceed the length limit. On the other hand, for long texts, even if they are related to the query question, they are generally not completely relevant. Chunking can eliminate irrelevant content to a certain extent and filter some for subsequent reply generation. Unnecessary noise.
The quality of text segmentation will greatly affect the effect of subsequent reply generation. If the segmentation is not good, the correlation between the content will be cut off. Therefore it is important to design a good chunking strategy. Chunking strategies include specific segmentation methods (such as whether to segment by sentences or paragraphs), the appropriate size of the blocks, whether overlap between different blocks is allowed, etc. Pinecone's blog [Chunking Strategies for LLM Applications](https://www.pinecone.io/learn/chunking-strategies/) gives some factors to consider when designing chunking strategies.
- **Characteristics of original content**: Is the original content long (blog posts, books, etc.) or short (tweets, instant messages, etc.), what format is it (HTML, Markdown, Code, or LaTeX, etc.), different content Features may lend themselves to different chunking strategies;
- **Subsequent indexing method**: The most commonly used index at present is vector indexing of the divided content, so different vector embedding models may have their own applicable block sizes, such as **sentence-transformer** The model is more suitable for embedding sentence-level content. The suitable block size of OpenAI's **text-embedding-ada-002** model is between 256 and 512 tokens;
- **Question Length**: The length of the question needs to be considered, because relevant text fragments need to be retrieved based on the question;
- **How ​​to use the retrieved relevant content in the reply generation phase**: If the retrieved relevant content is directly provided to LLM as part of the Prompt, then the input length limit of LLM needs to be considered when designing the block size. 

#### Block implementation method
So how to implement text blocking? Generally speaking, the overall process of implementing text chunking is as follows:
1. Divide the original long text into small semantic units, where the semantic units are usually at the sentence level or paragraph level;
2. Fuse these small semantic units into larger chunks until the set chunk size (Chunk Size) is reached, then treat the chunk as an independent text fragment;
3. Iteratively construct the next text fragment. Generally, there will be overlap between adjacent text fragments to maintain semantic coherence.
So how to split the original long text into small semantic units? The most commonly used method is to split based on delimiters, such as periods (.), newlines (\\n), spaces, etc. In addition to using a single separator for simple segmentation, you can also define a set of separators for iterative segmentation, such as defining a set of separators such as `["\n\n", "\n", " ", ""]` character, when segmenting, first use the first separator to perform segmentation (to achieve an effect similar to segmenting by paragraph). After the first segmentation is completed, for blocks that exceed the preset size, continue to use subsequent separators. Slice, and so on. This segmentation method can better maintain the hierarchical structure of the original text.
For some structured texts, such as code, Markdown, LaTeX and other texts, you may need to consider them separately when segmenting:
- For example, in Python code files, you may need to add characters like `\nclass` and `\ndef` in the delimiter to ensure the integrity of class and function code blocks;
- For example, Markdown files are organized through different levels of Headers, that is, different numbers of \# symbols. This hierarchical structure can be maintained by using specific separators when splitting.
The setting of text block size is also an important factor to consider in the chunking strategy. If it is too large or too small, it will affect the effect of final reply generation. The most commonly used calculation method for text block size can be based directly on the number of characters (Character-level) or based on the number of tokens (Token-level). As for how to determine the appropriate block size, this varies from scenario to scene, and it is difficult to have a unified standard. The choice can be made by evaluating the effects of different block sizes.
Some of the blocking methods mentioned above have corresponding implementations in [LangChain](https://python.langchain.com/docs/modules/data_connection/document_transformers/). For example, the following code example

```python
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

# text split
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 100,
    chunk_overlap  = 20,
    length_function = len,
    add_start_index = True,
)

# code split
python_splitter = RecursiveCharacterTextSplitter.from_language(
						language=Language.PYTHON, 
						chunk_size=50, 
						chunk_overlap=0  
)

# markdown split
md_splitter = RecursiveCharacterTextSplitter.from_language(  
						language=Language.MARKDOWN, 
						chunk_size=60, 
						chunk_overlap=0  
)

```

### Data index
![list_query](assets/list_query.webp)
![list_filter_query](assets/list_filter_query.webp)
#### Tree index
A tree index builds a set of nodes (text blocks) into a hierarchical tree-like index structure that builds upward from leaf nodes (original text blocks), with each parent node being a summary of a child node. In the retrieval phase, you can either traverse downward from the root node, or directly use the information of the root node. Tree indexes provide a more efficient way to query long blocks of text, and can also be used to extract information from different parts of the text. Unlike chained indexes, tree indexes do not require sequential queries.
![tree_index](assets/tree_index.webp)
![tree_query](assets/tree_query.png)
#### Keyword table index
The keyword table index extracts keywords from each node and constructs a many-to-many mapping of each keyword to the corresponding node, which means that each keyword may point to multiple nodes, and each node may also contain multiple keywords. . During the retrieval phase, nodes can be filtered based on keywords in user queries.
![keyword_table_index](assets/keyword_table_index.webp)
![keyword_query](assets/keyword_query.webp)
#### Vector index
Vector indexing is currently the most popular indexing method. This method generally uses the **Text Embedding Model** (Text Embedding Model) to map the text block into a fixed-length vector, and then stores it in the **Vector Database**. During retrieval, the user query text is mapped into a vector using the same text embedding model, and then the most similar node or nodes are obtained based on vector similarity calculation.
![vector_store_index](assets/vector_store_index.webp)
![vector_store_query](assets/vector_store_query.webp)
The above statement involves three important concepts in vector indexing and retrieval: **text embedding model**, **similar vector retrieval** and **vector database**. The details are explained one by one below.
##### Text embedding model
Text Embedding Model converts unstructured text into structured vectors (Vectors). Currently, dense vectors obtained through learning are commonly used.
![vectors-2](assets/vectors-2.svg)
There are currently many text embedding models to choose from, such as
- Early Word2Vec, GloVe models, etc., are rarely used at present.
- [Sentence Transformers](https://arxiv.org/abs/1908.10084) model based on twin BERT network pre-training, which has better embedding effect on sentences
- [text-embedding-ada-002](https://openai.com/blog/new-and-improved-embedding-model) model provided by OpenAI, the embedding effect is good and can handle text with a maximum length of 8191 tokens
- [Instructor](https://instructor-embedding.github.io/) model, which is an instruction-fine-tuned text embedding model that can be customized based on tasks (such as classification, retrieval, clustering, text evaluation, etc.) and domains ( Such as science, finance, etc.), providing task instructions to generate relatively customized text embedding vectors without any fine-tuning.
- [BGE](https://github.com/FlagOpen/FlagEmbedding/blob/master/README_zh.md) model: Chinese and English semantic vector model open sourced by Zhiyuan Research Institute, currently ranked at MTEB in both Chinese and English lists The first one.
The following is the list [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for evaluating the effect of text embedding models (as of 2023-08-18). It is worth noting that these off-the-shelf text embedding models are not fine-tuned for specific downstream tasks, so they may not necessarily perform well enough on downstream tasks. The best way is generally to retrain or fine-tune your own text embedding model on specific data downstream.
![mteb_leaderboard_20230816](assets/mteb_leaderboard_20230816.png)
##### Similar vector retrieval
The problem to be solved by similar vector retrieval is that given a query vector, how to accurately and efficiently retrieve one or more vectors that are similar to it from candidate vectors. The first is the choice of **similarity measurement** method, which can use cosine similarity, dot product, Euclidean distance, Hamming distance, etc. Under normal circumstances, cosine similarity can be used directly. The second is the choice of similarity retrieval algorithm and implementation method. The magnitude of candidate vectors, retrieval speed and accuracy requirements, memory limitations, etc. are all factors that need to be considered.
When the number of candidate vectors is relatively small, for example, there are only tens of thousands of vectors, then the Numpy library can implement similar vector retrieval, which is simple to implement, highly accurate, and fast. A foreign blogger did a simple benchmark test and found that [Do you actually need a vector database](https://www.ethanrosenthal.com/2023/04/10/nn-vs-ann/), when the candidate vector When the number is below 100,000, by comparing Numpy with another efficient approximate nearest neighbor retrieval implementation library [Hnswlib](https://github.com/nmslib/hnswlib), we found that there is no order of magnitude difference in retrieval efficiency. Differences, but Numpy's implementation is simpler.
![hnsw_numpy_nn_search_benchmark](assets/hnsw_numpy_nn_search_benchmark.png)
The following is a simple implementation code using Numpy:
```python
import numpy as np

# candidate_vecs: 2D numpy array of shape N x D
# query_vec: 1D numpy array of shape D
# k: number of top k similar vectors

sim_scores = np.dot(candidate_vecs, query_vec)
topk_indices = np.argsort(sim_scores)[::-1][:k]
topk_values = sim_scores[topk_indices]
```

For similarity retrieval of large-scale vectors, it is not appropriate to use the Numpy library, and a more efficient implementation solution is needed. [Faiss](https://github.com/facebookresearch/faiss), which is open sourced by the Facebook team, is a good choice. Faiss is a library for efficient similarity search and vector clustering. It implements many algorithms for searching in vector collections of any size. In addition to running on the CPU, some algorithms also support GPU acceleration. Faiss contains a variety of similarity retrieval algorithms. Which algorithm to use depends on factors such as data volume, retrieval frequency, accuracy, and retrieval speed.
This blog by Pinecone [Nearest Neighbor Indexes for Similarity Search](https://www.pinecone.io/learn/series/faiss/vector-indexes/) provides a detailed introduction to several indexes commonly used in Faiss, as shown below It is a qualitative comparison of several indexes in different dimensions:
![faiss_ann_algo_comparison](assets/faiss_ann_algo_comparison.png)
##### Vector database
The vector similarity retrieval scheme based on Numpy and Faiss mentioned above may still lack some functions if applied to actual products, such as:
- Data hosting and backup
- Data management, such as data insertion, deletion and update
- Storage of original data and metadata corresponding to vectors
- Scalability, including vertical and horizontal expansion
So **vector database** came into being. Simply put, a vector database is a database specifically used to store, manage and query vector data, and can achieve similar retrieval, clustering, etc. of vector data. The more popular vector databases currently include [Pinecone](https://www.pinecone.io/), [Vespa](https://vespa.ai/), [Weaviate](https://weaviate.io/) ,[Milvus](https://milvus.io/),[Chroma](https://www.trychroma.com/) ,[Tencent Cloud VectorDB](https://cloud.tencent.com/product/vdb ), etc., most of which provide open source products.
Pinecone's blog [What is a Vector Database](https://www.pinecone.io/learn/vector-database/) provides a relatively systematic introduction to the relevant principles and composition of vector databases. The following picture This is a common data processing process for vector databases given in the article:
![vector_database_pipeline](assets/vector_database_pipeline.png)
1. **Index**: Index vectors using algorithms such as Product Quantization (Product Quantization), Locality Sensitive Hash (LSH), HNSW, etc. This step maps the vector to a data structure to achieve faster search.
2. **Query**: Compare the query vector and the index vector to find the nearest neighbor similar vector.
3. **Post-processing**: In some cases, after the nearest neighbor vector is retrieved from the vector database, it is post-processed and then the final result is returned.
The use of the vector database is relatively simple. The following is a sample code for using Python to operate the Pinecone vector database:

```python
# install python pinecone client
# pip install pinecone-client
import pinecone 
# initialize pinecone client
pinecone.init(api_key="YOUR_API_KEY", environment="YOUR_ENVIRONMENT")
# create index 
pinecone.create_index("quickstart", dimension=8, metric="euclidean")
# connect to the index
index = pinecone.Index("quickstart")
# Upsert sample data (5 8-dimensional vectors) 
index.upsert([ 
			  ("A", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), 
			  ("B", [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]), 
			  ("C", [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]), 
			  ("D", [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]), 
			  ("E", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) 
			])

# query
index.query( 
			vector=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], 
			top_k=3, 
			include_values=True 
			) 

# Returns: 
# {'matches': [{'id': 'C', 
#               'score': 0.0, 
#               'values': [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]}, 
#              {'id': 'D', 
#               'score': 0.0799999237, 
#               'values': [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]}, 
#              {'id': 'B', 
#               'score': 0.0800000429, 
#               'values': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]}], 
# 'namespace': ''}

# delete index 
pinecone.delete_index("quickstart")
```

## Query and retrieval module
Relevant studies have shown that ([self-ask](https://ofir.io/self-ask.pdf), [ReAct](https://arxiv.org/abs/2210.03629) ), LLM is more effective in answering complex questions. , if complex questions are decomposed into relatively simple sub-questions, the response performance will be better. Here it can be divided into **single-step decomposition** and **multi-step decomposition**.
**Single-step decomposition** converts a complex query into multiple simple subqueries, and fuses the answer to each subquery as a reply to the original complex query.
![single_step_diagram](assets/single_step_diagram.png)
For **multi-step decomposition**, given an initial complex query, it will be converted into multiple subqueries step by step, and the next step of query questions will be generated based on the response results of the previous step until no more questions can be asked. Finally, the responses from each step are combined to generate the final result.
![multi_step_diagram](assets/multi_step_diagram.png)
#### Transformation 3: HyDE
[HyDE](http://boston.lti.cs.cmu.edu/luyug/HyDE/HyDE.pdf), the full name is Hypothetical Document Embeddings. Given an initial query, LLM is first used to generate a hypothetical document or reply, and then Retrieve this hypothetical document or reply as a new query, rather than using the original query directly. This transformation may generate a misleading hypothetical document or response without context, which may result in an incorrect response that is not relevant to the original query. Here is an example given in the paper:
![hyde](assets/hyde.png)
### Sorting and post-processing
After the previous retrieval process, you may get a lot of related documents, which need to be filtered and sorted. Common filtering and sorting strategies include:
- Filter and sort based on similarity score
- Filter based on keywords, such as limiting inclusion or exclusion of certain keywords
- Let LLM reorder based on returned relevant documents and their relevance scores
- Filter and sort based on time, such as only filtering the latest relevant documents
- Weight similarity based on time, then sort and filter
## Reply generation module
### Reply generation strategy
The retrieval module retrieves relevant text chunks based on the user query, and the reply generation module lets LLM use the retrieved relevant information to generate a reply to the original query. There are some different reply generation strategies given in LlamaIndex.
One strategy is to combine each retrieved relevant text chunk sequentially, continuously revising the generated responses each time. In this case, there will be as many LLM calls as there are independent relevant blocks of text. Another strategy is to populate the Prompt with as many blocks of text as possible on each LLM call. If one Prompt cannot be filled, use similar operations to construct multiple Prompts. The calls to multiple Prompts can use the same reply correction strategy as the previous one.
### Reply to generate Prompt template
Below is a Prompt template provided in LlamaIndex that generates responses. As you can see from this template, you can use some delimiters (such as ------) to distinguish the text of relevant information. You can also specify whether LLM needs to combine its own knowledge to generate a reply, and when providing relevant information. If it doesn’t help, do you want to reply or not?

```python
template = f'''
Context information is below.
---------------------
{context_str}
---------------------
Using both the context information and also using your own knowledge, answer the question: {query_str}

If the context isn't helpful, you can/don’t answer the question on your own.
'''
```

The Prompt template below allows LLM to continuously revise existing responses.

```python
template = f'''
The original question is as follows: {query_str}
We have provided an existing answer: {existing_answer}
We have the opportunity to refine the existing answer (only if needed) with some more context below.
------------
{context_str}
------------
Using both the new context and your own knowledege, update or repeat the existing answer.
'''
```

# Case analysis and application
[Github Copilot](https://github.com/features/copilot) is an AI-assisted programming tool. If you have used it, you will find that Github Copilot can help users automatically generate or complete code according to the context of the code. Sometimes you may just write down the class name or function name, or after writing the function comment, Copilot will give you the generated code. code, and many times it may be the code we want to implement. Since Github Copilot is not open source, some people on the Internet have conducted reverse engineering analysis of its VSCode plug-in, such as [copilot internals](https://thakkarparth007.github.io/copilot-explorer/posts/copilot-internals) and [copilot analysis](https ://github.com/mengjian-github/copilot-analysis), so that we can have a general understanding of the internal implementation of Copilot. To put it simply, the Github Copilot plug-in will collect the user's various contextual information in the VSCode programming environment to construct a Prompt, and then send the constructed Prompt to the code generation model (such as Codex) to obtain the completed code and display it in the editor. middle. How to retrieve relevant context information (Context) is a very important link. Github Copilot is an application of search-enhanced LLM in the direction of AI-assisted programming.
It should be noted that the two reverse engineering analyzes mentioned above were done a few months ago. Github Copilpot may have done a lot of updates and iterations. In addition, the analysis was obtained by the original author after reading and understanding the reversed code, so it may be There will be some deviations in understanding. The following content is generated by me combining those two analyses, so some places may be inaccurate or even wrong, but it does not prevent us from using the example of Copilot to understand the importance of contextual information in enhancing LLM output results, and to learn some Practical ideas for context-sensitive information retrieval.
The following is an example of a Prompt. You can see that it contains prefix code information (prefix), suffix code information (suffix), generation mode (isFimEnabled), and starting position information of different elements of the Prompt (promptElementRanges).
![github_copilot_prompt_example](assets/github_copilot_prompt_example.png)
Regardless of the effect of the code generation model itself, the quality of Prompt construction will largely affect the effect of code completion, and the extraction and composition of context-related information (Context) will largely determine the quality of Prompt construction. bad. Let’s take a look at some key ideas and implementations of context-sensitive information extraction in Github Copilot’s Prompt construct.
Copilot's Prompt contains different types of related information, including
- `BeforeCursor`: the content before the cursor
- `AfterCursor`: content after the cursor
- `SimilarFile`: code snippets that are highly similar to the current file
- `ImportedFile`: import dependencies
- `LanguageMarker`: language mark at the beginning of the file
- `PathMarker`: relative path information of the file
To extract similar code snippets, multiple recently accessed files in the same language will first be obtained as candidate documents for extracting similar code snippets. Then set the window size (for example, the default is 60 lines) and the step size (for example, the default is 1 line), and divide the candidate documents into code blocks in a sliding window manner. Then calculate the similarity between each segmented code block and the current file, and finally retain several code blocks with higher similarity. The acquisition of the current file here is to intercept the content of the window size from the current cursor forward. The similarity measure uses the **Jaccard coefficient**. Specifically, each line in the code block will be segmented and common codes will be filtered. Keywords (such as if, then, else, for these), get a token (Token) set, and then calculate the Jaccard similarity between the Token set of the current code block and the candidate code block. In the Copilot scenario, this similarity calculation method is simple and effective.
$$J(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \ cap B|}$$
The above analysis article summarizes the composition of Prompt into the following picture.
![github_copilot_prompt_components](assets/github_copilot_prompt_components.png)
After constructing the Prompt, Copilot will also determine whether it is necessary to initiate a request. The calculation of the code generation model is very computationally intensive, so it is necessary to filter some unnecessary requests. One of the judgments is to use a simple linear regression model to score the Prompt. When the score is lower than a certain threshold, the request will not be issued. This linear regression model utilizes features such as code language, whether the last code completion suggestion was accepted or rejected, the length of time since the last code completion suggestion was accepted or rejected, the character to the left of the cursor, etc. By analyzing the weights of the model, the original author made some observations:
- Some programming languages ​​have a higher weight than other languages ​​(php > js > python > rust > ...), PHP has the highest weight, and sure enough **PHP is the best language in the world** ( ^\_^ ).
- It is logical that the right half bracket (e.g. `)`, `]` ) has less weight than the left half bracket.
Through the analysis of Github Copilot, a programming aid, we can see:
- Retrieval-enhanced LLM ideas and technologies play an important role in the implementation of Github Copilot
- Context-related information (Context) can be a broad concept, it can be related text or code fragments, or it can be file paths, related dependencies, etc. Each scenario can define its specific context elements.
- The measurement of similarity and the similar retrieval method can vary depending on the scenario. Not all scenarios need to use cosine similarity. They all need to find relevant documents through vector similarity retrieval. For example, the implementation of Copilot uses a simple Jaccard coefficient. To calculate the similarity of the Token set after word segmentation, it is simple and efficient.
## Retrieval and Q&A of documents and knowledge bases

A typical application of retrieval-enhanced LLM technology is knowledge base or document question and answer, such as retrieval and question and answer for an enterprise's internal knowledge base or some documents. There are currently many commercial and open source products in this application direction. For example, [Mendable](https://www.mendable.ai/) is a commercial product that provides document-based AI retrieval and question and answer capabilities. The search capabilities for the official documents of the LlamaIndex and LangChain projects mentioned above are provided by Mendable. Below is a screenshot of usage. You can see that Mendable will not only give generated responses, but also attach reference links.
![mendable_screenshot](assets/mendable_screenshot.png)
In addition to commercial products, there are also many similar open source products. for example
- [Danswer](https://github.com/danswer-ai/danswer): Provides a question and answer function for internal corporate documents, can import data from multiple sources, supports traditional retrieval and LLM-based question and answer, and can intelligently Identify the user's search intent, thereby adopting different retrieval strategies, supporting user and document permission management, and supporting Docker deployment, etc.
- [PandaGPT](https://www.pandagpt.io/): Supports users to upload files and then ask questions about the file content
- [FastGPT](https://fastgpt.run/): An open source LLM-based AI knowledge base Q&A platform
- [Quivr](https://github.com/StanGirard/quivr): This open source project enables users to search and question personal files or knowledge bases, hoping to become the user's "second brain"
- [ChatFiles](https://github.com/guangzhengli/ChatFiles): Another LLM-based document Q&A open source project
The picture below is the technical architecture diagram of the ChatFiles project. It can be found that the basic modules and architecture of such projects are very similar. They basically follow the idea of ​​​​retrieval-enhanced LLM. This type of knowledge base question and answer application has almost become the **Hello World in the LLM field.** Applied.

![chatfiles_architecture](assets/chatfiles_architecture.png)


# References

1. [ChatGPT Retrieval Plugin](https://github.com/openai/chatgpt-retrieval-plugin) #project 
2. [Hypothetical Document Embeddings](https://arxiv.org/abs/2212.10496?ref=mattboegner.com) #paper
3. [Knowledge Retrieval Architecture for LLM’s (2023)](https://mattboegner.com/knowledge-retrieval-architecture-for-llms/) #blog
4. [Chunking Strategies for LLM Applications](https://www.pinecone.io/learn/chunking-strategies/) #blog
5. [LangChain Document Transformers](https://python.langchain.com/docs/modules/data_connection/document_transformers/) #doc
6. [LlamaIndex Index Guide](https://gpt-index.readthedocs.io/en/latest/core_modules/data_modules/index/index_guide.html) #doc
7. [Full stack LLM Bootcamp: Augmented Language Models](https://fullstackdeeplearning.com/llm-bootcamp/spring-2023/augmented-language-models/) #course
8. [Pinecone: vector indexes in faiss](https://www.pinecone.io/learn/series/faiss/vector-indexes/) #blog 
9. [Pinecone: what is a vector database](https://www.pinecone.io/learn/vector-database/) #blog 
10. [Zero and Few Shot Text Retrieval and Ranking Using Large Language Models](https://blog.reachsumit.com/posts/2023/03/llm-for-text-ranking/) #blog 
11. [copilot internals](https://thakkarparth007.github.io/copilot-explorer/posts/copilot-internals) #blog 
12. [copilot analysis](https://github.com/mengjian-github/copilot-analysis) #blog 
13. [Discover LlamaIndex: Key Components to Build QA Systems](https://www.youtube.com/watch?v=A3iqOJHBQhM&ab_channel=LlamaIndex) #video 
14. [Billion scale approximate nearest neighbor search](https://wangzwhu.github.io/home/file/acmmm-t-part3-ann.pdf) #slide
15. [ACL 2023 Tutorial: Retrieval based LM](https://acl2023-retrieval-lm.github.io/) #slide
16. [Pinecone: why use retrieval instead of larger context](https://www.pinecone.io/blog/why-use-retrieval-instead-of-larger-context/) #blog 
17. [RETA-LLM](https://github.com/RUC-GSAI/YuLan-IR/tree/main/RETA-LLM) #project
18. [Document Metadata and Local Models for Better, Faster Retrieval](https://www.youtube.com/watch?v=njzB6fm0U8g&ab_channel=LlamaIndex) #video
