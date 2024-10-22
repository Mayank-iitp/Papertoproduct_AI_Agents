# Research Summarization and Product Innovation Pipeline

This project is designed to automate a complete workflow from **research paper analysis** to **innovative product conceptualization** and **market analysis**. It utilizes Langchain agents and tools to process research papers, generate product ideas, evaluate technical feasibility, and assess market viability—all within a structured pipeline.

## Overview

The pipeline simulates a team of expert agents, each with specific roles:

1. **Research Summarizer**: Extracts key findings from research papers.
2. **Product Innovator**: Develops innovative product ideas based on the research findings.
3. **Technical Feasibility Assessor**: Evaluates the technical feasibility of the proposed products.
4. **Market Intelligence Specialist**: Analyzes market trends and competitors to validate commercial potential.
5. **Strategic Communications Expert**: Synthesizes all findings into a final report.

The pipeline runs each task in sequence, simulating how a real-world team might collaborate to turn research insights into actionable strategies.

## Key Features

- **PDF Processing**: Automatically processes research papers in PDF format.
- **Internet Search Integration**: Uses real-time search to gather market insights.
- **Agent Collaboration**: Agents can delegate tasks, mimicking real-world collaboration.
- **Structured Workflows**: Tools and agents follow a clear, structured process from research analysis to report generation.

---

## Project Structure

- **Agents**: Experts in different fields (researcher, innovator, market analyst, etc.) who handle specific tasks.
- **Tools**: Internet search and document processing tools aid in retrieving necessary information.
- **Tasks**: Each task is designed for a specific agent with a clear goal and output.
- **Crew**: A collection of agents working together in a sequential process to complete the project.

---
## Sample workflow

![Screenshot 2024-10-21 123244](https://github.com/user-attachments/assets/a55e245a-94c1-4747-9d43-33e3a4fa58d7)


---


## How It Works

1. **Process a Research Paper**: The system loads and processes a PDF, splitting it into chunks for easy retrieval.
2. **Summarize Key Findings**: The `Research Summarizer` agent extracts key findings from the research paper.
3. **Generate Product Ideas**: The `Product Innovator` agent creates innovative product ideas based on the research findings.
4. **Evaluate Technical Feasibility**: The `Technical Feasibility Assessor` ranks the ideas based on their technical viability.
5. **Conduct Market Analysis**: The `Market Intelligence Specialist` performs market research to assess the commercial potential of the top product ideas.
6. **Generate Final Report**: The `Strategic Communications Expert` compiles all findings into an action-oriented report.

---

## Setup Instructions

Step 1: Clone the repo 

Step 2: create .env file and put Hugging face token and groq api key
```
HF_TOKEN="YOUR HF TOKEN"
GROQ_API_KEY="YOUR GROQ API KEY"

```
Step 3: Download required packages
```
pip install -r requirements.txt
```
Step 4: run the main file
```
python main.py
```


