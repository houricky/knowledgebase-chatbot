from configparser import ConfigParser
import json
from typing import Dict, Any
import boto3

# config.ini namespaces
DB_CREDENTIALS = 'postgresql_credentials'
AZURE_CREDENTIALS = 'azure_credentials'

conf_obj= ConfigParser()
conf_obj.read('conf/config.ini')

def get_postgres_config() -> Dict[str, Any]:
    """
        retrieve postgres configuration from configuration file
        :return: Dict of configuration key value pairs
    """
    dbname = conf_obj.get(DB_CREDENTIALS, "DATABASE")
    user = conf_obj.get(DB_CREDENTIALS,"USER")
    password = conf_obj.get(DB_CREDENTIALS,"PASSWORD")
    host = conf_obj.get(DB_CREDENTIALS,"HOST")
    port = conf_obj.get(DB_CREDENTIALS,"PORT")
    return {
        "dbname" : dbname,
        "user" : user,
        "password" : password,
        "host" : host,
        "port" : port,
        "connection_string":  f"postgresql://{user}:{password}@{host}:{port}/{dbname}", 
    }


def get_azure_api_config() -> Dict[str, Any]: 
    """
        load azure api config from configuration file
        :return: Dict of configuration key value pairs
    """
    api_key = get_openai_api_key()
    if api_key is None:
        api_key = conf_obj.get(AZURE_CREDENTIALS, 'OPENAI_API_KEY')

    return {
        "openai_api_key" : api_key,
        "openai_deployment_name" : conf_obj.get(AZURE_CREDENTIALS, 'OPENAI_DEPLOYMENT_NAME'),
        "openai_embedding_model_name": conf_obj.get(AZURE_CREDENTIALS, 'OPENAI_EMBEDDING_MODEL_NAME'),
        "openai_api_base": conf_obj.get(AZURE_CREDENTIALS, 'OPENAI_API_BASE'),
        "model_name" : conf_obj.get(AZURE_CREDENTIALS, 'MODEL_NAME'),
        "openai_api_version" : conf_obj.get(AZURE_CREDENTIALS, 'OPENAI_API_VERSION'),
        "temperature" : conf_obj.get('model_setting', 'TEMPERATURE'),
     
        }


def get_openai_api_key() -> Dict[str, Any]:
    """
        retrive the Azure OpenAI key from the AWS secret manager
        :return: Dict of secret values
    """
    try:
        secret = get_secret()
        secret_dict = json.loads(secret)

        return secret_dict['OPENAI_API_KEY']
    except Exception as exception:
        print(f"Error retrieving OpenAI API Key  secret: {exception}")   
        return None

if __name__ == "__main__":
    db_config = get_postgres_config()
    azure_config = get_azure_api_config()
    
