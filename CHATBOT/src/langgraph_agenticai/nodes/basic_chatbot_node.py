from src.langgraph_agenticai.state.state import State

class BasicChatbotNode:
    
    def __init__(self,model): #what is model here it is confusing me 
        self.llm =model
        
    def process(self,state:State)->dict:
        """
        Processes the input state and generates a chatbot response
        """
        return {"messages":self.llm.invoke(state['messages'])}
        