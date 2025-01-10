# from typing import Dict, List, Tuple, Any
# from langgraph.graph import StateGraph, Graph
# from langchain_community.chat_models import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema import HumanMessage, AIMessage
# from app.schemas.chat import AgentState, ChatResponse
# from app.models.dynamodb import LeadInformation
# import os
# import uuid

# class ChatAgentService3:
#     def __init__(self):
#         self.llm = ChatOpenAI(
#             model="gpt-4-turbo-preview",
#             temperature=0,
#             api_key=os.getenv("OPENAI_API_KEY")
#         )
#         self.graph = self._create_graph()

#     def _create_graph(self) -> Graph:
#         workflow = StateGraph(AgentState)

#         # Define the nodes with synchronous functions
#         # workflow.add_node("classify_intent", self._classify_intent)
#         # workflow.add_node("route_to_agent", self._route_to_agent)
#         workflow.add_node("handle_customer_service", self._handle_customer_service)
#         # workflow.add_node("handle_sales", self._handle_sales)
#         # workflow.add_node("handle_technical", self._handle_technical)

#         # Define the edges
#         # workflow.add_edge("classify_intent", "route_to_agent")
#         # workflow.add_conditional_edges(
#         #     # "route_to_agent",
#         #     self._route_condition,
#         #     {
#         #         "customer_service": "handle_customer_service",
#         #         # "sales": "handle_sales",
#         #         # "technical": "handle_technical",
#         #     }
#         # )

#         # Set entry point
#         workflow.set_entry_point("handle_customer_service")

#         return workflow.compile()

#     def _classify_intent(self, state: AgentState) -> AgentState:
#         messages = state.messages
#         if not messages:
#             return state

#         classify_prompt = ChatPromptTemplate.from_messages([
#             ("system", "You are a helpful assistant that classifies customer inquiries into categories: customer_service, sales, or technical."),
#             ("human", "{message}")
#         ])

#         last_message = messages[-1]["content"] if messages else ""
#         response = self.llm.invoke(classify_prompt.format_messages(message=last_message))
        
#         state.context["intent"] = response.content.lower().strip()
#         return state

#     def _route_to_agent(self, state: AgentState) -> AgentState:
#         state.current_status = "routing"
#         return state

#     def _route_condition(self, state: AgentState) -> str:
#         return state.context.get("intent", "customer_service")

#     def _handle_customer_service(self, state: AgentState) -> AgentState:
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", "You are a helpful customer service representative. Help resolve customer issues professionally and courteously."),
#             ("human", "{message}")
#         ])
        
#         last_message = state.messages[-1]["content"] if state.messages else ""
#         response = self.llm.invoke(prompt.format_messages(message=last_message))
        
#         state.messages.append({"role": "assistant", "content": response.content})
#         state.current_status = "customer_service_handled"
#         return state

#     def _handle_sales(self, state: AgentState) -> AgentState:
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", "You are a helpful sales representative. Focus on understanding customer needs and providing relevant product information."),
#             ("human", "{message}")
#         ])
        
#         last_message = state.messages[-1]["content"] if state.messages else ""
#         response = self.llm.invoke(prompt.format_messages(message=last_message))
        
#         state.messages.append({"role": "assistant", "content": response.content})
#         state.current_status = "sales_handled"
#         return state

#     def _handle_technical(self, state: AgentState) -> AgentState:
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", "You are a helpful technical support representative. Provide clear, step-by-step solutions to technical issues."),
#             ("human", "{message}")
#         ])
        
#         last_message = state.messages[-1]["content"] if state.messages else ""
#         response = self.llm.invoke(prompt.format_messages(message=last_message))
        
#         state.messages.append({"role": "assistant", "content": response.content})
#         state.current_status = "technical_handled"
#         return state

#     async def process_message(self, session_id: str, message: str, user_id: str) -> ChatResponse:
#         # Initialize or get existing state
#         state = AgentState(
#             messages=[{"role": "user", "content": message}],
#             current_status="started",
#             user_id=user_id,
#             session_id=session_id or str(uuid.uuid4())
#         )

#         # Process the message through the graph
#         # Use invoke instead of await since the graph uses synchronous functions
#         final_state = self.graph.invoke(state)

#         # Get the last assistant message
#         last_message = next(
#             (msg["content"] for msg in reversed(final_state.messages) 
#              if msg["role"] == "assistant"),
#             "No response generated."
#         )

#         return ChatResponse(
#             response=last_message,
#             session_id=final_state.session_id,
#             status=final_state.current_status
#         )

