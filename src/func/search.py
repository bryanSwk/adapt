import os
import json

#Search Libraries
from googlesearch import search
import arxiv

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import BrowserlessLoader

browserless_api_key = os.getenv("BROWSERLESS_KEY")

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                 model_kwargs={'device': 'cuda'})

def summary(query, url):
    print("Starting summary..")
    info, source = [], []
    loader = BrowserlessLoader(api_token=browserless_api_key, urls=url)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    all_splits = text_splitter.split_documents(documents)       
    db = FAISS.from_documents(all_splits, embedder)
    docs = db.similarity_search(query, 5)
    for doc in docs:
        info.append(doc.page_content)
        source.append(doc.metadata['source'])
    return info, source
                                 

def search_google(query, num_results=5):
    search_results = search(query, sleep_interval=5, num_results=num_results)
    info, source = summary(query, search_results)
    return info, source


def search_arxiv(query, max_results=5):
    search = arxiv.Search(
    query = query,
    max_results = max_results,
    sort_by = arxiv.SortCriterion.SubmittedDate
    )

    url = []

    for result in search.results():
        print(result.title)
        url.append(result.pdf_url)
    
    return url

