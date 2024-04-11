import os
from crewai import Agent, Task, Process, Crew

from langchain.agents import Tool
from langchain.agents import load_tools
from langchain.utilities import GoogleSerperAPIWrapper


# to get your api key for free, visit and signup: https://serper.dev/
os.environ["SERPER_API_KEY"] = "serp-api-here"

# Ollama
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_MODEL_NAME"] = "llama2:latest"
os.environ["OPENAI_API_KEY"] = "DUMMY_API_KEY"

search = GoogleSerperAPIWrapper()

search_tool = Tool(
    name="Scrape google searches",
    func=search.run,
    description="useful for when you need to ask the agent to search the internet",
)

# Loading Human Tools
human_tools = load_tools(["human"])

psychologist = Agent(
    role="Psychologist",
    goal="Provide insights and guidance for individuals seeking emotional well-being, personal development, and coping strategies.",
    backstory="""You are trained in understanding human behavior, emotions, and mental processes. Your expertise lies in providing support, guidance, and interventions to help individuals navigate challenges, improve relationships, and enhance overall well-being. Your goal is to empower individuals to lead fulfilling lives by addressing their psychological needs and fostering personal growth.""",
    verbose=True,
    allow_delegation=True,
)

life_coach = Agent(
    role="Life Coach",
    goal="Assist individuals in setting and achieving personal and professional goals, enhancing motivation, and overcoming obstacles.",
    backstory="""You are dedicated to helping individuals identify their values, strengths, and aspirations, and to develop actionable plans to achieve their desired outcomes. Your role is to inspire, motivate, and empower individuals to realize their full potential, overcome barriers, and create meaningful and fulfilling lives. Your approach is client-centered, collaborative, and focused on facilitating positive change and growth.""",
    verbose=True,
    allow_delegation=True,
)

advice_columnist = Agent(
    role="Advice Columnist",
    goal="Offer practical advice, insights, and perspectives on various life challenges, relationships, and personal dilemmas.",
    backstory="""You are skilled in providing guidance, wisdom, and support to individuals seeking advice on a wide range of topics, including relationships, career decisions, personal growth, and self-improvement. Your goal is to offer empathetic, insightful, and actionable recommendations that empower individuals to make informed choices and navigate life's complexities with confidence and resilience.""",
    verbose=True,
    allow_delegation=False,   
    tools=[search_tool],
)

def ask_question():
    question = input("Ask a question about life: ")
    return question

def process_question(question):
    taskPsy = Task(
        description=f"""Provide personalized guidance and coping strategies for managing stress and anxiety in daily life, based on the question: '{question}'. Write a detailed report with practical tips, exercises, and resources for building resilience and promoting emotional well-being.""",
        agent=psychologist,
        expected_output="Detailed report offering personalized strategies for managing stress and anxiety.",
    )
    taskCoach = Task(
        description=f"""Help individuals identify their core values, strengths, and aspirations, and develop a roadmap for achieving their personal and professional goals, based on the question: '{question}'. Write a detailed plan outlining actionable steps, milestones, and strategies for overcoming obstacles and staying motivated.""",
        agent=life_coach,
        expected_output="Detailed plan for achieving personal and professional goals.",
    )
    taskJournalist = Task(
        description=f"""Offer advice and insights on common life challenges, dilemmas, and relationship issues, based on the question: '{question}'.""",
        agent=advice_columnist,
        expected_output="Series of articles or blog posts offering practical advice and insights on life challenges and relationship issues.",
    )

    return [taskPsy, taskCoach, taskJournalist]


question = ask_question()
tasks = process_question(question)

crew = Crew(
    agents=[psychologist, life_coach, advice_columnist],
    tasks=tasks,
    verbose=2,
    process=Process.sequential,
    max_iterations=30
)

result = crew.kickoff()

print("######################")
print(result)
