# main.py contains the core function that loads your LangGraph + Streamlit UI.
# app.py is just a launcher script

import streamlit as st
from src.langgraph_agenticai.ui.streamlitui.loadui import LoadStreamlitUI
from src.langgraph_agenticai.llms.groqllm import GroqLLM
from src.langgraph_agenticai.graph.graph_builder import GraphBuilder

def load_langgraph_agenticai_app():
    """
    Loads and runs the LangGraph AgenticAI application with Streamlit UI.
    This function initializes the UI, handles user input, configures the LLM model,
    sets up the graph based on the selected use case, and displays the output while 
    implementing exception handling for robustness.

    """

    ui =LoadStreamlitUI()
    user_input =ui.load_streamlit_ui()

    if not user_input:
        st.error("Error: Failed to load user input from the UI.")
        return
    
    user_message = st.chat_input("Enter your message:")

    if user_message:
        try:
            # configure the LLM's
            obj_llm_config =GroqLLM(user_controls_input=user_input)
            model =obj_llm_config.get_llm_model()

            if not model:
                st.error('Error : LLM model could not be initialized')
                return 

            #Iniitalize and setup the graph based on usecase
            use_case = user_input.get('selected_usecase')

            if not use_case:
                st.error('Error: No use Case Selected .Please select Any one of the UseCase')
                return

            graph_builder =GraphBuilder(model)
            try:
                graph =graph_builder.setup_graph(usecase=use_case)
            except Exception as e:
                st.error(f"Error:Graph Setup Failed->{e}")
                return



        except Exception as e:
            return



    

