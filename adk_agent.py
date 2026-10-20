
import asyncio
import os
import sys

from dotenv import load_dotenv
from mcp import StdioServerParameters

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
from google.genai import types


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

APP_NAME = "aerointel"
USER_ID = "aerointel_user"
SESSION_ID = "aerointel_session"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MCP_SERVER_PATH = os.path.join(BASE_DIR, "mcp_server.py")


# --------------------------------------------------
# 2. Configure the existing MCP server
# --------------------------------------------------

server_params = StdioServerParameters(
    command=sys.executable,
    args=["-u", MCP_SERVER_PATH],
)

aviation_tools = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=server_params,
        timeout=60,
    )
)


# --------------------------------------------------
# 3. Configure the ADK agent
# --------------------------------------------------

agent = LlmAgent(
    name="aerointel_agent",
    model="gemini-3.8-flash",
    description="An aviation intelligence assistant.",
    instruction="""
You are AeroIntel, an aviation intelligence assistant.

Rules:
1. Use the available MCP tools to retrieve evidence
   before answering factual aviation questions.
2. For aircraft comparisons, retrieve information about
   each aircraft being compared.
3. For airline questions, use the relevant airline tool.
4. Base factual claims on retrieved information.
5. Never invent specifications, figures, or sources.
6. If information is missing, clearly state that it is
   not available in the retrieved records.
7. Explain differences clearly using headings or tables
   when appropriate.
""",
    tools=[aviation_tools],
)


# --------------------------------------------------
# 4. Run the agent
# --------------------------------------------------

async def main():
    session_service = InMemorySessionService()

    try:
        # Initialize the MCP connection and discover tools.
        print("Connecting AeroIntel ADK to MCP server...")

        tools = await aviation_tools.get_tools()

        print("\nMCP connection successful!")
        print("Available tools:")

        for tool in tools:
            print(f"- {tool.name}")

        # Create a conversation session.
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID,
        )

        # Create the ADK runner.
        runner = Runner(
            app_name=APP_NAME,
            agent=agent,
            session_service=session_service,
        )

        print("\nAeroIntel ADK is ready.")

        while True:
            question = input(
                "\nAsk AeroIntel (type 'exit' to quit): "
            ).strip()

            if question.lower() in {"exit", "quit"}:
                print("Shutting down AeroIntel...")
                break

            if not question:
                print("Please enter a question.")
                continue

            message = types.Content(
                role="user",
                parts=[types.Part(text=question)],
            )

            print("\n[ADK] Processing your question...\n")

            try:
                async for event in runner.run_async(
                    user_id=USER_ID,
                    session_id=SESSION_ID,
                    new_message=message,
                ):
                    if event.is_final_response() and event.content:
                        for part in event.content.parts or []:
                            if part.text:
                                print(part.text)

            except Exception as exc:
                print(f"Agent request failed: {exc}")

    except Exception as exc:
        print(f"\nAeroIntel startup failed: {exc}")

    finally:
        # Release the MCP connection.
        try:
            await aviation_tools.close()
        except Exception as exc:
            print(f"MCP cleanup warning: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
