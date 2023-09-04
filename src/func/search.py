import os
import json

#Search Libraries
from googlesearch import search
import arxiv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import BrowserlessLoader


def summary(query, url, embedder):
    print("Starting summary..")
    info, source = [], []
    print("Scraping Websites..")
    loader = BrowserlessLoader(api_token=browserless_api_key, 
                               urls=url)
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
    documents = loader.load_and_split(text_splitter=splitter)
    
    # all_splits = text_splitter.split_documents(documents)
    print("Creating Vector DB..")       
    db = FAISS.from_documents(documents, embedder)
    docs = db.similarity_search(query, 7)
    for doc in docs:
        info.append(doc.page_content)
        source.append(doc.metadata['source'])
    return info, set(source)
                                 

def search_google(query, embedder, num_results=3):
    search_results = search(query, sleep_interval=5, num_results=num_results)
    info, source = summary(query, search_results, embedder)
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

