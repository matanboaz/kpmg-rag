#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


# STEP 1: Install Libraries
get_ipython().system('pip install -qU langchain-community langchain-core openai pinecone tiktoken docx2txt langchain_pinecone')


# In[3]:


# STEP 2: Imports
import os
import time
from typing import List, Dict
from tqdm.auto import tqdm

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.schema import HumanMessage

from pinecone import Pinecone, ServerlessSpec


# In[4]:


# STEP 3: API Keys (insert your real keys here)
os.environ["OPENAI_API_KEY"] = "sk-..."  # Securely load via env
openai_api_key = os.environ["OPENAI_API_KEY"]
pinecone_api_key = "pcsk_..."  # Use from .env in production


# In[5]:


# STEP 4: Pinecone Init
pc = Pinecone(api_key=pinecone_api_key)
spec = ServerlessSpec(cloud="aws", region="us-east-1")
index_name = "rag-kpmg"

if index_name not in [i['name'] for i in pc.list_indexes()]:
    pc.create_index(index_name, dimension=1536, metric="dotproduct", spec=spec)
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

index = pc.Index(index_name)


# In[6]:


# STEP 5: Load and Chunk DOCX Files
from google.colab import files
uploaded = files.upload()  # Upload all 4 DOCX files here

def load_documents_from_docx(filenames):
    """
    Load DOCX documents, infer topics, and split into chunks for embedding.

    Parameters:
        filenames (list): List of uploaded document names

    Returns:
        List[dict]: Chunked and metadata-enriched document segments
    """
    all_chunks = []
    file_id = 0

    for filename in filenames:
        loader = Docx2txtLoader(filename)
        raw_docs = loader.load()
        text = raw_docs[0].page_content

        # Split content into overlapping chunks for embedding
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)

        # Infer topic from filename
        if "steel" in filename.lower():
            topic = "steel"
        elif "textile" in filename.lower():
            topic = "textile"
        elif "food" in filename.lower():
            topic = "food"
        elif "auto" in filename.lower():
            topic = "automobile"
        else:
            topic = "general"

        # Embed title and section headers into chunks
        for chunk_id, chunk in enumerate(chunks):
            section_hint = chunk.strip().split("\n")[0][:100]
            full_chunk = f"{filename} — {section_hint}\n{chunk}"
            all_chunks.append({
                'chunk': full_chunk,
                'source': filename,
                'title': filename,
                'topic': topic,
                'doi': str(file_id),
                'chunk-id': str(chunk_id)
            })

        file_id += 1

    return all_chunks

# documents = load_documents_from_uploads()
filenames = ["Steel.docx", "Textile.docx", "Food.docx", "Automobile.docx"]
documents = load_documents_from_docx(filenames)
df = pd.DataFrame(documents)


# In[7]:


# STEP 6: Embed + Upload to Pinecone
embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

batch_size = 100
for i in tqdm(range(0, len(df), batch_size)):
    batch = df.iloc[i:i+batch_size]
    ids = [f"{x['doi']}-{x['chunk-id']}" for _, x in batch.iterrows()]
    texts = batch["chunk"].tolist()
    embeddings = embed_model.embed_documents(texts)
    metadata = [{'text': x['chunk'], 'source': x['source'], 'title': x['title']} for _, x in batch.iterrows()]
    vectors = list(zip(ids, embeddings, metadata))
    index.upsert(vectors=vectors)


# In[8]:


# STEP 7: Initialize Vectorstore + Chat Model
text_field="text"
vectorstore = PineconeVectorStore(
    index, embed_model, text_field
    )
chat = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")


# In[15]:


# STEP 8: Utility Functions for Preprocessing and Prompting
def infer_topic_from_query(query):
    """
    Infer topic from a user query string.

    Parameters:
        query (str): User input

    Returns:
        str or None: One of {steel, textile, food, automobile} or None
    """
    q = query.lower()
    if "steel" in q: return "steel"
    if "textile" in q: return "textile"
    if "food" in q: return "food"
    if "automobile" in q or "car" in q or "vehicle" in q: return "automobile"
    return None

def preprocess_query(query: str) -> str:
    """
    Clean or clarify vague or short queries.

    Parameters:
        query (str): User input

    Returns:
        str: Augmented query string
    """
    q = query.strip().lower()

    # Handle single years
    if q.isdigit() and len(q) == 4:
        return f"{query} — please specify an industry or event you're interested in."

    # Handle overly short or vague inputs
    if len(q) < 6:
        return f"The query '{query}' is too short. Please ask a specific question related to a topic, event, or industry."

    # Fix vague "innovation" queries without time range
    if "innovation" in q and not any(t in q for t in ["20th", "21st", "post-war", "after 1900"]):
        return query + " in the 20th century"

    return query


# In[16]:


# STEP 9: Smart Prompt Generator
def augment_prompt(query: str, k: int = 8):
    """
    Build an augmented prompt using smart chunk retrieval and query-type awareness.

    Parameters:
        query (str): User input
        k (int): Number of top documents to retrieve

    Returns:
        str: Final prompt for the language model
    """
    q_lower = query.lower()
    topic = infer_topic_from_query(query)
    filters = {"topic": topic} if topic else None

    # Retrieve top-k most relevant chunks
    results = vectorstore.similarity_search(query, k=k, filter=filters)

    # Fallback: try without filter if no good results
    if not results or all(len(r.page_content.strip()) == 0 for r in results):
        results = vectorstore.similarity_search(query, k=k)
        if not results:
            return f"""No relevant information was found in the documents to answer the query: "{query}"."""

    # Merge results into context
    source_knowledge = "\n".join([r.page_content for r in results])

    # Select smart prompt based on query type
    if any(kw in q_lower for kw in ["why", "reason", "explain why"]):
        instruction = "Based on the context below, explain the reason clearly and completely."
    elif any(kw in q_lower for kw in ["define", "what is", "describe"]):
        instruction = "Based on the context below, give a clear and concise definition or description."
    elif any(kw in q_lower for kw in ["how", "process", "steps"]):
        instruction = "Based on the context below, explain how it works or how it evolved over time."
    else:
        instruction = "Using the context below, answer the query as accurately as possible."

    return f"""You are a helpful assistant. {instruction}

Context:
{source_knowledge}

Query: {query}
"""



edge_cases = {
    "Ambiguous": "What happened in 1945?",
    "Unanswerable": "What are the textile policies in Mars colony?",
    "Vague": "Tell me about changes.",
    "Multi-topic": "Compare post-war developments in food and steel manufacturing.",
    "Misspelled": "What is post-waar foood scociety?",
    "Very long": "Can you give a full explanation of technological shifts in food, textile, and steel industries over time?",
    "Short": "1950",
    "Similar wording": "Post-war steel developments?",
    "Rephrased 1": "Describe the post-war period in food manufacturing.",
    "Rephrased 2": "How did food production evolve after WWII?",
    "Contradictory Premise 1": "Which food companies were privatized in 1900?",
    "Contradictory Premise 2": "How did Tesla innovate in textile manufacturing?",
    "Casing & Punctuation": "wHaT Is FoRdIsM???",
    "Formatting Variation": "textile-industrialization timeline",
    "Temporal Ambiguity 1": "How did production change over time?",
    "Temporal Ambiguity 2": "What were the most important innovations?",
    "Partial Input 1": "Assembly lines?",
    "Partial Input 2": "EV battery production",
    "Cross-document": "Compare manufacturing innovations in steel and textiles from 1900 to 1950.",
    "Rare Entity 1": "What was Ransom Olds known for?",
    "Rare Entity 2": "Did Eiji Toyoda influence global manufacturing?",
    "Why Question": "Why did lean manufacturing outperform Fordism?",
    "How Question": "How did the COVID-19 pandemic affect supply chains in food and auto industries?",
}


for label, query in edge_cases.items():
    print(f"\n🟪 {label} — Query: {query}")
    print("🔍 Answer:\n")
    try:
        adjusted_query = preprocess_query(query)
        prompt = augment_prompt(adjusted_query)
        response = chat.invoke([HumanMessage(content=prompt)])
        print(response.content.strip())
    except Exception as e:
        print(f"❌ Error: {e}")
    print("=" * 80)


# In[13]:


# DEBUGGING TOOL
def debug_retrieval(query: str, k: int = 10):
    """
    Debug retrieval by showing top-k matching chunks and similarity scores.

    Parameters:
        query (str): The user input
        k (int): Number of top chunks to retrieve

    Returns:
        None (prints output)
    """
    topic = infer_topic_from_query(query)
    filters = {"topic": topic} if topic else None

    print(f"\n🔍 Query: {query}")
    print(f"📎 Topic filter: {filters or 'None'}\n")

    results = vectorstore.similarity_search_with_score(query, k=k, filter=filters)

    if not results:
        print("❌ No relevant chunks found.")
        return

    for i, (doc, score) in enumerate(results):
        print(f"🔹 Rank #{i+1} | Score: {score:.3f}")
        print(doc.page_content[:300] + "...\n")

# Queries that underperformed and need debugging
edge_case_queries = {
    "Casing & Punctuation": "What is Fordism?",
    "Formatting": "textile-industrialization timeline",
    "Similar wording": "Post-war steel developments?",
    "Why Question": "Why did lean manufacturing outperform Fordism?"
}

# Evaluate each case using debug_retrieval
for label, query in edge_case_queries.items():
    print("=" * 100)
    print(f"🧪 DEBUG — {label} — Query: {query}")
    debug_retrieval(query, k=10)


# In[ ]:




