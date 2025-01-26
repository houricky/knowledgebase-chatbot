from langchain.vectorstores import Chroma
import psycopg2
from psycopg2 import sql, Error
import pandas as pd
from langchain.vectorstores.pgvector import PGVector
from utils.config_manager import get_config_by_key, get_postgres_config_secret
from llama_index.vector_stores import PGVectorStore
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.vector_stores import PGVectorStore

# initialize config variables
db_config = get_postgres_config_secret()

class PGVectorDb:
    def __init__(self, database_name,embeddings):
        self.connection_string = db_config['connection_string']
        self.collection_name = "Synopsis_KnowledgeBase"
        #self.collection_name = db_config['collection_name']
        self.embeddings = embeddings
    def database_reader(self):

        VectorpgDb = PGVector(
            collection_name=self.collection_name,
            connection_string=self.connection_string,
            embedding_function=self.embeddings)
        return VectorpgDb

class ChromaDb:
    def __init__(self, database_name,embeddings):
        self.database_directory= get_config_by_key(database_name, 'DATABASE_DIRECTORY')
        self.embeddings=embeddings

    def database_reader(self):
        """

        :return: chroma database object
        """
        vectorchromaDb = Chroma(
            persist_directory=self.database_directory,
            embedding_function=self.embeddings,
            # client_settings=Settings(anonymized_telemetry=False)
        )
        return vectorchromaDb
    def database_creator(self,splits):
        """
        :param splits:it contains the documents chunks
        :return: persist the created database
        """
        is_created=False
        try:
            vectorchromaDb = Chroma.from_documents(
                    documents=splits,
                    embedding=self.embeddings,
                    persist_directory=self.database_directory
                )
            vectorchromaDb.persist()
            is_created=True
        except Exception as e:
            print("The error is: ",e)
            is_created=False
        return is_created



class PostgreSQLCRUD:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname= db_config['dbname'],
            user = db_config['user'],
            password = db_config['password'],
            host = db_config['host'],
            port = db_config['port']
        )
        self.conn.autocommit = True

    def create_record(self, table_name, columns, values):
        '''
        :param table_name: name of the tables in which we want to insert the data
        :param columns: specific column names where we want to insert the data
        :param values: actual data to be inserted in the table for specific columns
        :return:
        '''
        try:
            with self.conn.cursor() as cursor:
                query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                    sql.Identifier(table_name),
                    sql.SQL(', ').join(map(sql.Identifier, columns)),
                    sql.SQL(', ').join(sql.Placeholder() * len(columns))
                )
                cursor.execute(query, values)
        except Error as e:
            print("Error:", e)

    def read_records(self, table_name, columns=None):
        '''
        :param table_name: name of the tables from which one want to extract data
        :param columns: specific columns one want to fetch
        :return:
        '''
        try:
            with self.conn.cursor() as cursor:
                if columns:
                    query = sql.SQL("SELECT {} FROM {}").format(
                        sql.SQL(', ').join(map(sql.Identifier, columns)),
                        sql.Identifier(table_name)
                    )
                else:
                    query = sql.SQL("SELECT * FROM {}").format(sql.Identifier(table_name))
                cursor.execute(query)
                rows = cursor.fetchall()
                return rows
        except Error as e:
            print("Error:", e)

    def read_records_with_filter(self, table_name, columns=None, condition_column=None, condition_value=None):
        '''
        :param table_name: name of the tables from which one want to extract data
        :param columns: specific columns one want to fetch
        :return:
        '''
        try:
            with self.conn.cursor() as cursor:
                if columns:
                    query = sql.SQL("SELECT {} FROM {} WHERE {} = %s").format(
                        sql.SQL(', ').join(map(sql.Identifier, columns)),
                        sql.Identifier(table_name),
                        sql.Identifier(condition_column)
                    )
                else:
                    query = sql.SQL("SELECT * FROM {}").format(sql.Identifier(table_name))
                cursor.execute(query,(condition_value,))
                rows = cursor.fetchall()
                rowDf = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
                return rowDf
        except Error as e:
            print("Error:", e)

    def update_record(self, table_name, update_column, update_value, condition_column, condition_value):
        '''
        :param table_name: name of the table name
        :param update_column: columns to be updated.
        :param update_value: actual value with which we want to update the columns
        :param condition_column: column on which filter to be applied
        :param condition_value: specific condition value
        :return:
        '''
        try:
            with self.conn.cursor() as cursor:
                query = sql.SQL("UPDATE {} SET {} = %s WHERE {} = %s").format(
                    sql.Identifier(table_name),
                    sql.Identifier(update_column),
                    sql.Identifier(condition_column)
                )
                cursor.execute(query, (update_value, condition_value))

        except Error as e:
            print("Error:", e)

    def delete_record(self, table_name, condition_column, condition_value):
        '''
        :param table_name: name of the table
        :param condition_column: column on which one want to filter
        :param condition_value: value for the particular column on which record to be deleted.
        :return:
        '''
        try:
            with self.conn.cursor() as cursor:
                query = sql.SQL("DELETE FROM {} WHERE {} = %s").format(
                    sql.Identifier(table_name),
                    sql.Identifier(condition_column)
                )
                cursor.execute(query, (condition_value,))
        except Error as e:
            print("Error:", e)

    def close_connection(self):
        '''
        used for closing the database connection
        :return:
        '''
        self.conn.close()

class Database:

    def __init__(self,datbase_name,embeddings=None):
        self.database_name = datbase_name
        self.embeddings=embeddings
        self.database_type = get_config_by_key(self.database_name, 'DATABASE_TYPE')

    def database_reader(self):
        """
        :self: self.conf_obj,self.database_name,self.embeddings
        :return: database reader
        """
        if self.database_type == "chroma":
            chroma_obj=ChromaDb(self.database_name,self.embeddings)
            return chroma_obj.database_reader()
        if self.database_type == "pg_vector":
            pgvector_obj = PGVectorDb(self.database_name,self.embeddings)
            return pgvector_obj.database_reader()

    def database_creator(self,splits):
        if self.database_type == "chroma":
            chroma_obj=ChromaDb(self.database_name,self.embeddings)
            return chroma_obj.database_creator(splits)


def vector_store_index(table_name):
    # Postgres database connection
    db_config = get_postgres_config_secret()
    db_name = db_config['dbname']
    db_user = db_config['user']
    db_password = db_config['password'] 
    db_host = db_config['host']
    db_port = db_config['port']

    # query pgvector index
    vector_store = PGVectorStore.from_params(
        database=db_name,
        host=db_host,
        password=db_password,
        port=db_port,
        user=db_user,
        table_name=table_name,
    )
    
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)   
    return index