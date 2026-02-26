from langchain_ollama import ChatOllama
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage
)

from tools import TOOLS

from dotenv import load_dotenv
import os


import logging
LOGGING_LEVEL = os.getenv('LOGGING_LEVEL', 'INFO')

formatter = logging.Formatter(fmt="[{levelname}] {asctime} :({name}): {message}",
                              datefmt="%Y-%m-%d %H:%M:%s", style="{")
file_handler = logging.FileHandler(filename="rag_agent_logs.log", mode="a", encoding='utf-8')
file_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(LOGGING_LEVEL)


load_dotenv()

MODEL = os.environ['MODEL']
SYSTEM_PROMPT = os.environ['SYSTEM_PROMPT']
MAX_ITERATIONS = int(os.environ['MAX_ITERATIONS'])

class RAGAgent:
    def __init__(self, tools) -> None:
        self.tools = {tool.name: tool for tool in tools}
        self.llm = ChatOllama(model=MODEL).bind_tools(tools)
        self.history: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT)]
        logger.info("Agent set up successfully")

    def _run_tool(self, tool_calls) -> list[ToolMessage]:
        tool_results = []
        for tool in tool_calls:
            tool_name, tool_args = tool['name'], tool['args']
            logger.info(f"Calling tool {tool_name}({(tool_args)})")
            try:
                tool_result = self.tools[tool_name].func(**tool_args)
            except Exception as e:
                logger.error(e, exc_info=True)
            tool_results.append(ToolMessage(content=tool_result, tool_call_id=tool['id']))
        return tool_results

    def ask(self, user_message: str):
        logger.info("Control over to AI.")
        self.history.append(HumanMessage(content=user_message))
        messages = self.history.copy()
        new_messages_index = len(messages)
        for i in range(MAX_ITERATIONS):
            logger.debug(f"Iteration {i}. {len(messages)=}, {new_messages_index=}")
            ai_message = AIMessage("")

            tools_to_use = []
            for ai_msg_chunk in self.llm.stream(messages):
                if ai_msg_chunk.tool_calls:
                    logger.info(f"Tool detected: {ai_msg_chunk.tool_calls}.")
                    tools_to_use.extend(ai_msg_chunk.tool_calls)
                ai_message.content += str(ai_msg_chunk.content)
                yield ai_msg_chunk.content

            logger.debug(f"AI message is {ai_message.content}. Tools to use: {[tool['name'] for tool in tools_to_use]}")
            if ai_message.content:
                messages.append(ai_message)
                break

            if tools_to_use:
                tool_messages = self._run_tool(tools_to_use)
                messages.extend(tool_messages)

        self.history.extend([m for m in messages[new_messages_index:] if isinstance(m, AIMessage)])
        

    def chat(self):
        print("\n🔬  Scientific RAG Agent ready. Type 'quit' to exit.")
        red, green, reset = "\033[31m", "\033[32m", "\033[0m"
        while True:
            try:
                user_input = input(f"{green}\nYou{reset}: ").strip()
            except EOFError:
                print("\nGoodbye!")
                break
            except KeyboardInterrupt:
                continue
            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break
            print(f"\n{red}Assistant{reset}: ", sep='', end='')
            for message_chunk in self.ask(user_input):
                print(message_chunk, sep='', end='', flush=True)
            print()


if __name__ == "__main__":

    agent = RAGAgent(tools=TOOLS)
    agent.chat()
