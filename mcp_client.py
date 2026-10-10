
import anyio

from mcp import Client, StdioServerParameters


server = StdioServerParameters(
    command=r"C:\AIprojects\AeroIntel\.venv\Scripts\python.exe",
    args=[r"C:\AIprojects\AeroIntel\mcp_server.py"],
)


async def main():
    async with Client(server) as client:

        # Discover the tools exposed by AeroIntel
        result = await client.list_tools()

        print("Available AeroIntel tools:")

        for tool in result.tools:
            print(f"- {tool.name}: {tool.description}")

        # Call the aircraft search tool
        result = await client.call_tool(
            "search_aircraft",
            {"aircraft": "A350-900"}
        )

        print("\nAircraft search result:")

        for item in result.content:
            if hasattr(item, "text"):
                print(item.text)


if __name__ == "__main__":
    anyio.run(main)
