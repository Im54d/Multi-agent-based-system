import json
import numpy as np
import pandas as pd
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.visualization.modules import CanvasGrid, TextElement
from mesa.visualization.ModularVisualization import ModularServer
from collections import deque
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os
import requests
class Message:
    def __init__(self, sender_id, receiver_id, content):
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.content = content

class EventElement(TextElement):
    def __init__(self):
        pass

    def render(self, model):
        return "Events:<br>" + "<br>".join(model.events[-5:])

class ResearchAgent(Agent):
    def __init__(self, unique_id, model, profile, publications, api_info):
        super().__init__(unique_id, model)
        self.profile = profile
        self.publications = publications
        self.api_info = api_info

    def step(self):
        self.move()
        self.provide_information()
        self.check_messages()
        self.fetch_data()

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def provide_information(self):
        event = f"Researcher {self.profile['name']} with interests {self.profile['interests']} and publications {self.publications}."
        self.model.add_event(event)

    def send_message(self, receiver_id, content):
        message = Message(self.unique_id, receiver_id, content)
        self.model.message_queue.append(message)

    def check_messages(self):
        for message in list(self.model.message_queue):
            if message.receiver_id == self.unique_id:
                self.receive_message(message)
                self.model.message_queue.remove(message)

    def receive_message(self, message):
        event = f"Researcher {self.profile['name']} received message: {message.content}"
        self.model.add_event(event)

    def fetch_data(self):
        try:
            response = requests.get(
                self.api_info['endpoint'],
                headers=self.api_info['headers'],
                params=self.api_info['params']
            )
            if response.status_code == 200:
                data = response.json()
                event = f"Researcher {self.profile['name']} fetched data: {data}"
                self.model.add_event(event)
            else:
                event = f"Researcher {self.profile['name']} failed to fetch data"
                self.model.add_event(event)
        except Exception as e:
            event = f"Researcher {self.profile['name']} encountered an error: {e}"
            self.model.add_event(event)

class AnalystAgent(Agent):
    def __init__(self, unique_id, model, current_trends, api_info):
        super().__init__(unique_id, model)
        self.current_trends = current_trends
        self.api_info = api_info
        self.research_data = []

    def step(self):
        self.move()
        self.analyze_trends()
        self.check_messages()
        self.fetch_trends()
        self.analyze_research_data()

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def analyze_trends(self):
        event = f"Analyzing current trends: {self.current_trends}."
        self.model.add_event(event)

    def fetch_trends(self):
        try:
            response = requests.get(
                self.api_info['endpoint'],
                headers=self.api_info['headers'],
                params=self.api_info['params']
            )
            if response.status_code == 200:
                data = response.json()
                event = f"Analyst fetched trends: {data}"
                self.model.add_event(event)
                self.current_trends = data  
            else:
                event = f"Analyst failed to fetch trends"
                self.model.add_event(event)
        except Exception as e:
            event = f"Analyst encountered an error while fetching trends: {e}"
            self.model.add_event(event)

    def send_message(self, receiver_id, content):
        message = Message(self.unique_id, receiver_id, content)
        self.model.message_queue.append(message)

    def check_messages(self):
        for message in list(self.model.message_queue):
            if message.receiver_id == self.unique_id:
                self.receive_message(message)
                self.model.message_queue.remove(message)

    def receive_message(self, message):
        if "research_data" in message.content:
            self.research_data.append(message.content["research_data"])
        event = f"Analyst received message: {message.content}"
        self.model.add_event(event)

    def analyze_research_data(self):
        if self.research_data:
            combined_data = " ".join(self.research_data)
            event = f"Analyst analyzing research data: {combined_data}"
            self.model.add_event(event)
            self.research_data.clear()


class RecommenderAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.recommendations = []

    def step(self):
        self.move()
        self.create_recommendations()
        self.check_messages()

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def create_recommendations(self):
        profiles = [agent.profile for agent in self.model.schedule.agents if isinstance(agent, ResearchAgent)]
        publications = [agent.publications for agent in self.model.schedule.agents if isinstance(agent, ResearchAgent)]

        documents = [" ".join(profile['interests']) + " " + " ".join(publications) for profile, publications in zip(profiles, publications)]

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

        if tfidf_matrix.shape[0] > 1:
            cosine_similarities = cosine_similarity(tfidf_matrix)

            for i, agent in enumerate(self.model.schedule.agents):
                if isinstance(agent, ResearchAgent):
                    similar_indices = cosine_similarities[i].argsort()[:-5:-1]
                    recommendations = [documents[j] for j in similar_indices if j != i]
                    recommendation_text = f"Recommendation for {agent.profile['name']}: {recommendations}"
                    self.model.add_event(recommendation_text)
                    self.send_message(agent.unique_id, recommendation_text)

    def send_message(self, receiver_id, content):
        message = Message(self.unique_id, receiver_id, content)
        self.model.message_queue.append(message)

    def check_messages(self):
        for message in list(self.model.message_queue):
            if message.receiver_id == self.unique_id:
                self.receive_message(message)
                self.model.message_queue.remove(message)

    def receive_message(self, message):
        event= f"Recommender received message: {message.content}"
        self.model.add_event(event)

class NotifierAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.notifications = []

    def step(self):
        self.move()
        self.notify_researchers()
        self.check_messages()

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def notify_researchers(self):
        for agent in self.model.schedule.agents:
            if isinstance(agent, ResearchAgent):
                notification = f"Notification for {agent.profile['name']}: New event in {agent.profile['interests'][0]}."
                self.notifications.append(notification)
                self.model.add_event(notification)
                self.send_message(agent.unique_id, notification)

    def send_message(self, receiver_id, content):
        message = Message(self.unique_id, receiver_id, content)
        self.model.message_queue.append(message)

    def check_messages(self):
        for message in list(self.model.message_queue):
            if message.receiver_id == self.unique_id:
                self.receive_message(message)
                self.model.message_queue.remove(message)

    def receive_message(self, message):
        event = f"Notifier received message: {message.content}"
        self.model.add_event(event)

class MultiAgentSystem(Model):
    def __init__(self, json_file):
        super().__init__()
        self.grid = MultiGrid(10, 10, True)
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
                        agent_reporters={"Profile": lambda a: getattr(a, "profile", None)}
        )
        self.events = []
        self.message_queue = deque()  

        try:
            with open(json_file) as f:
                data = json.load(f)
        except FileNotFoundError:
            print("File agents_data.json not found.")
            return
        except json.JSONDecodeError:
            print("Error decoding JSON file.")
            return

        for i, agent_data in enumerate(data['agents']):
            agent_type = agent_data['type']
            api_info = agent_data.get('api', {})

            if agent_type == 'ResearchAgent':
                agent = ResearchAgent(i, self, agent_data['profile'], agent_data['publications'], api_info)
            elif agent_type == 'AnalystAgent':
                agent = AnalystAgent(i, self, agent_data['current_trends'], api_info)
            elif agent_type == 'RecommenderAgent':
                agent = RecommenderAgent(i, self)
            elif agent_type == 'NotifierAgent':
                agent = NotifierAgent(i, self)
            else:
                continue

            self.schedule.add(agent)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))
            self.server = None

    
        def stop_server(self):
        
            if self.server and not self.server.running:
            
                self.write_events_to_file("events.txt")
            
            self.server.close()

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)

    def add_event(self, event):
        self.events.append(event)
        if len(self.events) > 5:
            self.events.pop(0)

    def write_events_to_file(self, filename):
        try:
            
            print("Current working directory:", os.getcwd())
            
            with open(filename, "w") as file:
                for event in self.events:
                    file.write(event + "\n")
            print("Events successfully written to", filename)
        except Exception as e:
            print("Error writing events to file:", e)

def agent_portrayal(agent):
    portrayal = {
        "Shape": "circle",
        "Filled": "true",
        "r": 0.5,
        "Layer": 0,
    }

    if isinstance(agent, ResearchAgent):
        portrayal["Color"] = "blue"
        portrayal["Layer"] = 1
        portrayal["text"] = agent.profile['name']
        portrayal["text_color"] = "white"
    elif isinstance(agent, AnalystAgent):
        portrayal["Color"] = "green"
    elif isinstance(agent, RecommenderAgent):
        portrayal["Color"] = "red"
    elif isinstance(agent, NotifierAgent):
        portrayal["Color"] = "yellow"

    return portrayal

grid = CanvasGrid(agent_portrayal, 10, 10, 500, 500)
event_element = EventElement()

server = ModularServer(
    MultiAgentSystem,
    [grid, event_element],
    "Multi Agent System",
    {"json_file": "agents_data.json"},
)


server.model.server = server

server.port = 8521
server.launch()

