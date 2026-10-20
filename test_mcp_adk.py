
import asyncio
import os
import sys

from mcp import StdioServerParameters
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams


async def main():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-u", os.path.abspath("mcp_server.py")],
    )

    toolset = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=server_params,
            timeout=60,
        )
    )

    try:
        tools = await toolset.get_tools()
        print("\nTools discovered by ADK:")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
    finally:
        await toolset.close()


if __name__ == "__main__":
    asyncio.run(main())
