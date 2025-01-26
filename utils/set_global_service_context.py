from llama_index import LangchainEmbedding, ServiceContext, set_global_service_context
from src.llm_models import AzureOpenAI

def set_llm_service_context(  llm_type ):

    if llm_type != 'AzureOpenAI':
        return False # handle the case where another LLM is selected
    
    llm_obj = AzureOpenAI()
    llm = llm_obj.azure_open_ai_gpt35()
    embedding = llm_obj.azure_open_ai_embedding_gpt35()

    # special handling for compatibility between LLamaIndex and AzureOpenAI
    embedding_llm = LangchainEmbedding(
        embedding,
        embed_batch_size=1,
    )

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embedding_llm,

    )

    set_global_service_context(service_context)
    return True
