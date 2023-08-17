import json
import requests
from bs4 import BeautifulSoup
#Search Libraries
from googlesearch import search
import arxiv

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import BrowserlessLoader

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                 model_kwargs={'device': 'cuda'})

def summary(query, url):
    print("Start summary..")
    loader = BrowserlessLoader(api_token='d15ecd5b-c5cc-47e0-a8c9-872ba903097c', urls=url)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    all_splits = text_splitter.split_documents(documents)       
    db = FAISS.from_documents(all_splits, embedder)
    docs = db.similarity_search(query) 
                                 

def search_google(query, num_results=5):
    search_results = search(query, sleep_interval=5, num_results=num_results)
    vectdb = summary(query, list(search_results))


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

