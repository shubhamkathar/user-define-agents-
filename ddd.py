from crewai import Agent, Task, Crew, LLM
from crewai.process import Process
from crewai_tools import SerperDevTool
import os
from datetime import datetime
# Initialize your Ollama LLM instance
'''
ollama_llm = LLM(
    model='ollama/llama3.1',
    base_url='http://localhost:11434',
)
'''
nvidia_llm= LLM (
    model="deepseek-ai/deepseek-r1",
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = "00000000000"
)

# Initialize the SerperDevTool for internet searching (focused on Google News)
serper_api_key = os.getenv("SERPER_API_KEY")  # Ensure you have the API key stored in environment variable
serper_tool = SerperDevTool(
    search_url="https://news.google.com/home?hl=en-IN&gl=IN&ceid=IN:en",  # Google News URL
    api_key=serper_api_key,
    n_results=5  # Adjust the number of results as needed
)

# User input for agents and tasks
num_agents = int(input("How many agents do you want to create? "))
agents = []
tasks = []

for i in range(1, num_agents + 1):
    print(f"Creating Agent {i}")
    agent_name = input("Enter a name for the agent: ")
    role = input("Enter the role of the agent: ")
    goal = input("Enter the goal of the agent: ")
    backstory = input("Enter the backstory of the agent: ")

    agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=nvidia_llm,  # Use Ollama LLM here
        allow_delegation=True,
        verbose=True
    )
    agents.append((agent_name, agent))

    task_name = input(f"Enter a task name for Agent {agent_name}: ")
    description = input("Enter the task description: ")
    expected_output = input("Enter the expected output: ")

    # Set up the execution function to fetch real-time news (using the SerperDevTool)
    execution_fn = lambda: serper_tool.run(search_query="latest news")  # Default behavior for fetching news
    
    task = Task(
        description=description,
        agent=agent,
        expected_output=expected_output,
        execution_fn=execution_fn  # Set the task to fetch news
    )
    tasks.append((task_name, task))

# Create the crew
tagents_list = [agent for _, agent in agents]
tasks_list = [task for _, task in tasks]

my_crew = Crew(
    agents=tagents_list,
    tasks=tasks_list,
    verbose=1,
    process=Process.sequential
)

# Run the crew
result = my_crew.kickoff()
print(result)