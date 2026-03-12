from src.langgraph_agenticai.state.state import State
from src.langgraph_agenticai.nodes.basic_chatbot_node import BasicChatbotNode

from langgraph.graph import StateGraph,START,END

class GraphBuilder:
    def __init__(self,model):
        self.llm =model
        self.graph_builder =StateGraph(State)

    def basic_chatbot_build_graph(self):
        """
        Builds a basic chatbot graph using LangGraph
        This method initializes a chatbot node using the `BasicChatbotNode` class 
        and integrates it into the graph. The chatbot node is set as both the 
        entry and exit point of the graph.
        """
        self.basic_chatbot_node = BasicChatbotNode(self.llm)

        self.graph_builder.add_node('chatbot',self.basic_chatbot_node.process)
        self.graph_builder.add_node(START,'chatbot')
        self.graph_builder.add_node('chatbot',END)
    
    def setup_graph(self,usecase:str):
        """
        Sets up the Graph for selected Use Case
        """
        if usecase == "Basic Chatbot":
            self.basic_chatbot_build_graph()