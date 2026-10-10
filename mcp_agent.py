
import anyio

from dotenv import load_dotenv
from mcp import Client, StdioServerParameters
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

server = StdioServerParameters(
    command=r"C:\AIprojects\AeroIntel\.venv\Scripts\python.exe",
    args=[r"C:\AIprojects\AeroIntel\mcp_server.py"],
)


async def main():
    # Connect to the AeroIntel MCP server
    async with Client(server) as mcp_client:

        # Tool 1: Search aircraft information
        @tool
        async def search_aircraft(aircraft: str) -> str:
            """Search AeroIntel for information about a specific aircraft."""

            print("\n[AGENT] Selected tool: search_aircraft")
            print(f"[AGENT] Arguments: aircraft={aircraft!r}")

            result = await mcp_client.call_tool(
                "search_aircraft",
                {"aircraft": aircraft},
            )

            print("[MCP] search_aircraft call completed")

            if result.is_error:
                raise RuntimeError(
                    f"search_aircraft failed: {result.content}"
                )

            return "\n\n".join(
                block.text
                for block in result.content
                if hasattr(block, "text")
            )

        # Tool 2: Search airline information
        @tool
        async def search_airline(airline: str) -> str:
            """Search AeroIntel for information about a specific airline."""

            print("\n[AGENT] Selected tool: search_airline")
            print(f"[AGENT] Arguments: airline={airline!r}")

            result = await mcp_client.call_tool(
                "search_airline",
                {"airline": airline},
            )

            print("[MCP] search_airline call completed")

            if result.is_error:
                raise RuntimeError(
                    f"search_airline failed: {result.content}"
                )

            return "\n\n".join(
                block.text
                for block in result.content
                if hasattr(block, "text")
            )

        # Tool 3: General aviation knowledge search
        @tool
        async def search_aviation_knowledge(query: str) -> str:
            """Search AeroIntel for general aviation information."""

            print("\n[AGENT] Selected tool: search_aviation_knowledge")
            print(f"[AGENT] Arguments: query={query!r}")

            result = await mcp_client.call_tool(
                "search_aviation_knowledge",
                {"query": query},
            )

            print("[MCP] search_aviation_knowledge call completed")

            if result.is_error:
                raise RuntimeError(
                    f"search_aviation_knowledge failed: {result.content}"
                )

            return "\n\n".join(
                block.text
                for block in result.content
                if hasattr(block, "text")
            )

        # Configure the language model
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.8-flash"
        )

        # Create the agent
        agent = create_agent(
            model=llm,
            tools=[
                search_aircraft,
                search_airline,
                search_aviation_knowledge,
            ],
            system_prompt=(
                "You are AeroIntel, an aviation intelligence assistant. "
                "Use the available tools to retrieve evidence before answering "
                "factual aviation questions. "
                "For comparisons, retrieve information about each relevant "
                "aircraft or airline. "
                "Treat tool results as your evidence source. "
                "Do not invent specifications, numbers, or sources. "
                "If a requested fact is absent from the retrieved information, "
                "state that it is not specified in the available records. "
                "Distinguish typical values from maximum values and keep "
                "measurement units and comparison baselines consistent. "
                "For numerical comparisons, verify the values and arithmetic "
                "before stating the conclusion. "
                "If the retrieved evidence is contradictory or insufficient, "
                "explain the limitation instead of guessing."
            ),
        )

        # Get the user's question
        question = input("Ask AeroIntel: ").strip()

        if not question:
            print("Please enter a question.")
            return

        # Run the agent
        print("\n[AGENT] Processing your question...")

        result = await agent.ainvoke({
            "messages": [
                {"role": "user", "content": question}
            ]
        })

        # Display the final answer
        print("\nAeroIntel:")

        final_message = result["messages"][-1]
        content = final_message.content

        if isinstance(content, str):
            print(content)

        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        print(block.get("text", ""))
                elif hasattr(block, "text"):
                    print(block.text)
        else:
            print(content)


if __name__ == "__main__":
    anyio.run(main)
