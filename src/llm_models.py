from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from utils.config_manager import get_azure_api_config
import langchain

langchain.verbose = False

class AzureOpenAI:
    def __init__(self):
        """
        :param conf_obj:
        :return:
        """
        self.azure_config = get_azure_api_config()


    def azure_open_ai_gpt35(self):
        """

        :return: openAI model based on the parameters
        """
       
        llm = AzureChatOpenAI(openai_api_type="azure",
                              openai_api_base=self.azure_config["openai_api_base"],
                              openai_api_version=self.azure_config["openai_api_version"],
                              openai_api_key=self.azure_config["openai_api_key"],
                              deployment_name=self.azure_config["openai_deployment_name"],
                              model=self.azure_config["model_name"],
                              temperature=self.azure_config["temperature"])

        return llm

    def azure_open_ai_gpt(self):
        """

        :return: dictionary of openai models
        """
        open_ai_gpt_models = {"gpt-4-turbo":self.azure_open_ai_gpt4_turbo(),"gpt-3.5-turbo-16k":self.azure_open_ai_gpt35()}
        return open_ai_gpt_models

    def azure_open_ai_embedding_gpt35(self):
        """

        :return: embeddings from azure openAI
        """
        
        embeddings = OpenAIEmbeddings(
                            openai_api_key=self.azure_config["openai_api_key"],
                            openai_api_base=self.azure_config["openai_api_base"],
                            model=self.azure_config["openai_embedding_model_name"], 
                            deployment=self.azure_config["openai_embedding_model_name"],
                            openai_api_type='azure',
                            chunk_size=16, 
                            disallowed_special=() )
        return embeddings


