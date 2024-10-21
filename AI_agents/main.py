from crewai import Agent, Task, Crew, Process
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
from langchain.tools import StructuredTool
from pydantic import BaseModel
import os 
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
groq_api_key= os.environ["GROQ_API_KEY"]
llm= ChatGroq(
    api_key=groq_api_key,
    model_name= "groq/llama3-8b-8192"
)

# Tools
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embeddings)
    return db
class SearchInput(BaseModel):
    query: str 
search_tool = StructuredTool(
    name="Internet Search",
    func=DuckDuckGoSearchRun().run,  # Function that performs the search
    description="Search the internet for up-to-date information on market trends, technologies, and industry developments.",
    args_schema=SearchInput  # Specify the structured input schema
)

# Assume we have processed the PDF and created a retriever
pdf_db = process_pdf("C:/TechMeet/2410.05779v1.pdf")
retriever = pdf_db.as_retriever()
class ResearchAnalysisInput(BaseModel):
    query: str

research_tool = StructuredTool(
    name="Research Paper Analysis",
    func=retriever.get_relevant_documents,  
    description="Retrieve and analyze specific sections of the research paper to extract key findings and innovative concepts.",
    args_schema=ResearchAnalysisInput  
)

# Agents with enhanced prompts
researcher = Agent(
    role='Research Summarizer',
    goal='Extract and summarize groundbreaking findings from the research paper with a focus on commercial potential',
    backstory="""You are a cutting-edge research analyst with a keen eye for transformative scientific discoveries. 
    Your expertise lies in distilling complex academic papers into actionable insights for product development. 
    You have a track record of identifying research breakthroughs that have led to successful commercial applications.""",
    tools=[research_tool],
    max_iter=10,
    verbose=True,
    llm=llm
)

innovator = Agent(
    role='Product Innovator',
    goal='Conceptualize revolutionary products or services based on the research findings',
    backstory="""You are a visionary product strategist known for turning abstract scientific concepts into 
    game-changing market offerings. Your ability to see beyond the obvious has resulted in the creation of 
    several unicorn startups. You excel at identifying unique value propositions that can disrupt industries.""",
    tools=[search_tool],
    allow_delegation=True,
    verbose=True,
    llm=llm
)

tech_assessor = Agent(
    role='Technical Feasibility Assessor',
    goal='Evaluate the technical viability and potential challenges in developing the proposed products',
    backstory="""You are a seasoned CTO with extensive experience in bringing cutting-edge technologies to market. 
    Your expertise spans various fields including AI, biotechnology, and advanced materials. You have a knack for 
    identifying critical technical hurdles and innovative solutions to overcome them.""",
    tools=[search_tool],
    verbose=True,
    llm=llm
)

market_researcher = Agent(
    role='Market Intelligence Specialist',
    goal='Conduct in-depth market analysis to validate the commercial potential of the proposed products',
    backstory="""You are a renowned market analyst with a history of accurately predicting market trends and 
    identifying lucrative opportunities. Your insights have guided numerous companies to successfully position 
    their innovative products in competitive markets. You have a vast network of industry contacts and access 
    to exclusive market data.""",
    tools=[search_tool],
    verbose=True,
    llm=llm
)

report_writer = Agent(
    role='Strategic Communications Expert',
    goal='Synthesize all findings into a compelling, action-oriented report for high-level decision makers',
    backstory="""You are a master communicator with a talent for translating complex technical and market 
    information into clear, persuasive narratives. Your reports have been instrumental in securing funding 
    for groundbreaking projects and guiding corporate strategy. You excel at presenting information in a way 
    that inspires action and drives decision-making.""",
    verbose=True,
    llm=llm
)

# Tasks with more specific objectives
task1 = Task(
    description="""Thoroughly analyze the research paper and extract key findings with high commercial potential. 
    Focus on novel technologies, unexpected results, and insights that challenge current industry paradigms. 
    Provide a concise summary highlighting the most promising areas for product development.""",
    agent=researcher,
    expected_output="A detailed summary of key research findings with high commercial potential."
)

task2 = Task(
    description="""Based on the research summary, conceptualize 3-5 innovative product or service ideas. 
    For each idea, outline its unique value proposition, potential target market, and how it leverages the 
    core research findings. Think boldly – consider ideas that could create new market categories.""",
    agent=innovator,
    expected_output="A list of 3-5 innovative product or service ideas with detailed descriptions."
)



task4 = Task(
    description="""Perform comprehensive market research for the top 2-3 most feasible product ideas. 
    Analyze market size, growth potential, competitor landscape, and potential barriers to entry. 
    Identify key customer segments and potential early adopters. Provide market entry strategy recommendations.""",
    agent=market_researcher,
    expected_output="A comprehensive market analysis report for the top 2-3 product ideas."
)

task5 = Task(
    description="""Compile all findings into a strategic, action-oriented report. Synthesize the research insights, 
    product ideas, technical assessments, and market analysis into a cohesive narrative. Conclude with clear 
    recommendations for next steps, including suggested priorities for product development and market entry. 
    Ensure the report is compelling and tailored for executive-level decision makers.""",
    agent=report_writer,
    expected_output="A comprehensive strategic report synthesizing all findings and recommendations."
)

# Crew
crew = Crew(
    agents=[researcher, innovator, tech_assessor, market_researcher, report_writer],
    tasks=[task1, task2, task3, task4, task5],
    process=Process.sequential
)

# Run the crew
result = crew.kickoff()

print(result)