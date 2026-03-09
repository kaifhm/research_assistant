from rag_agent import RAGAgent
from tools import TOOLS

agent = RAGAgent(tools=TOOLS)
agent.chat()