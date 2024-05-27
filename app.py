from ss_llm.utils.SS_document_helpers import SS_load_documents
from ss_llm.utils.SS_classes import  SS_Models 
from ss_llm.utils.SS_Functions import get_answer_using_context, gpu_clean
from ss_llm.utils.SS_Custom_Embeddings import NomicCustomEmbeddingFunction, SS_SentenceTransformerEmbeddings
from ss_llm.services.local_datastore import SS_Store2VectorDB
from ss_llm.services.retriver_vectordb import RAG_retrived_docs
import torch


torch.device("cuda")
gpu_clean()

policies_path = "Resources"
embeded_splits = SS_load_documents(chunk_size=1024).files_to_docs(dir_path=policies_path)

#'''
### OLLAMA
ss_ollama_models = SS_Models()
embeddings_model = ss_ollama_models.set_ollama_embeddings("nomic-embed-text:latest")
llm = ss_ollama_models.set_ollama_llm("llama3:latest")
ss_ollama_models.model_info()
#''' 

''' 
### HuggingFace
ss_hf_models = SS_Models()
llm = ss_hf_models.set_huggingface_llm(model_name="microsoft/Phi-3-mini-4k-instruct")
embeddings_model = ss_hf_models.set_huggingFace_embeddings(model_name="nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
ss_hf_models.model_info()
'''  
 
#SS_Store2VectorDB().save2chromadb_using_embeddings(model=embeddings_model, docs=embeded_splits, save_path="VectoreStore/policy_DB",collection_name="ss_policies")
SS_Store2VectorDB().save2chromadb_using_embeddingsfunction(docs=embeded_splits,save_path="VectoreStore/policy_DB", custom_embeding=NomicCustomEmbeddingFunction() ,collection_name="ss_policies", isfull_load=True)


#''' 
question = "Where is Srihari Gonela working?"
#retrived_docs = RAG_retrived_docs().load_retriver_chromaDB(db_path="VectoreStore/policy_DB", embeddings = embeddings_model, question=question, collection_name="ss_policies")
retrived_docs = RAG_retrived_docs().load_ssearch_chromadb(
    db_path="VectoreStore/policy_DB", embeddings = embeddings_model, question=question, collection_name="ss_policies")

answer = get_answer_using_context(context=retrived_docs, question=question, llm=llm)
print("\n\n\n\n\nAnswwer is: \n"+answer)
  
#'''

print("Completed !")
