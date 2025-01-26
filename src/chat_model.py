from langchain.prompts.prompt import PromptTemplate
from langchain import LLMChain
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMMathChain
from langchain.agents import initialize_agent,AgentType

import langchain
langchain.verbose = False
def chat_bot(llm):
    template = """The following is a friendly conversation between a human and an AI.\
    The AI is talkative and provides lots of specific details from its context.\
    If the AI does not know the answer to a question, it truthfully says it does not know.
    
    Current conversation:
    {history}
    Human: {input}
    AI Assistant:"""

    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
    conversation = LLMChain(
        prompt=PROMPT,
        llm=llm,
        verbose=True,
    )
    return conversation

def _handle_error(error) -> str:
    return str(error)[:50]

def smart_chat_bot(llm,SERPER_API_KEY):
    memory = ConversationBufferMemory(memory_key="chat_history")
    search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    tools = [
        Tool(
            name="current_search",
            func=search.run,
            description="useful for when you need to answer questions about current events or the current state of state of the world or the real-time information",
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math"
        )
    ]

    agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,max_iterations=3,early_stopping_method="generate",memory=memory,handle_parsing_errors=_handle_error, verbose=True)
    return agent_chain