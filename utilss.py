# load core modules
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
import os
# load agents and tools modules
import pandas as pd
#from azure.storage.filedatalake import DataLakeServiceClient
from io import StringIO
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain import LLMMathChain

os.environ["OPENAI_API_KEY"] = "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"

# initialize pinecone client and connect to pinecone index
pinecone.init(
        api_key="f2d1d06d-e15d-433d-9979-7d6919724402",  
        environment="asia-southeast1-gcp-free"  
) 

index_name ='hello'
index = pinecone.Index(index_name) # connect to pinecone index

# initialize embeddings object; for use with user query/input
embed = OpenAIEmbeddings(
                model = 'text-embedding-ada-002',
                openai_api_key="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",
            )

# initialize langchain vectorstore(pinecone) object
text_field = 'text' # key of dict that stores the text metadata in the index
vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

llm = ChatOpenAI(    
    openai_api_key="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", 
    model_name="gpt-3.5-turbo", 
    temperature=0.0
    )

# initialize vectorstore retriever object
hello = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

df = pd.read_csv("C:\\Users\\Admin\\Desktop\\hello\\StudentDetailsComplete.csv") 
python = PythonAstREPLTool(locals={"df": df}) # set access of python_repl tool to the dataframe

# create calculator tool
calculator = LLMMathChain.from_llm(llm=llm, verbose=True)

# create variables for f strings embedded in the prompts
user = 'abhi' # set user
df_columns = df.columns.to_list() # print column names of df

# prep the (tk policy) vectordb retriever, the python_repl(with df access) and langchain calculator as tools for the agent
tools = [
    Tool(
        name = "hello",
        func=hello.run,
        description="""
        useful when you need to answer questions about esprit private university in general
        <user>: what is esprit 
        <assistant>: it's a private university in tunisia 
       
        """
    ),
    Tool(
        name = "students Data",
        func=python.run,
        description = f"""
        Useful for when you need to answer questions about StudentDetails data stored in pandas dataframe 'df'. 
        Run python pandas operations on 'df' to help you get the right answer.
        'df' has the following columns: {df_columns}
        
        <user>: what is my mark in sub1?
        <assistant>: df[df['name'] == '{user}']['sub1']
                     
        """
    ),
    Tool(
        name = "Calculator",
        func=calculator.run,
        description = f"""
        Useful when you need to do math operations or arithmetic.
        """
    )
]

# change the value of the prefix argument in the initialize_agent function. This will overwrite the default prompt template of the zero shot agent type
agent_kwargs = {'prefix': f'You are friendly HR assistant. You are tasked to assist the current user: {user} on questions related to HR. You have access to the following tools:'}


# initialize the LLM agent
agent = initialize_agent(tools, 
                         llm, 
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                         verbose=True, 
                         agent_kwargs=agent_kwargs
                         )
# define q and a function for frontend
def get_response(user_input):
    response = agent.run(user_input)
    return response