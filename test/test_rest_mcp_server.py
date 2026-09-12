import asyncio
import pydantic

from fastmcp import FastMCP

mcp = FastMCP("Python MCP Server")

@mcp.tool(description = "查询某个城市的出行方式")
def get_transport(city: str = pydantic.Field(description = "城市名称")) -> str:
    mock = {
        "北京": "如果晴天就步行，下雨就坐地铁，如果阴天就坐公交。",
        "广州": "如果晴天就步行，下雨就坐地铁，如果阴天就坐公交。",
        "深圳": "如果晴天就步行，下雨就坐地铁，如果阴天就坐公交。",
    }
    return mock[city] if city in mock else "自驾"

@mcp.tool(description = "查询城市天气")
def get_weather(city: str = pydantic.Field(description = "城市名称")) -> str:
    mock = {
        "北京": "晴天",
        "广州": "阴天",
        "深圳": "雨天",
    }
    return mock[city] if city in mock else "未知天气"
    
@mcp.resource("config://version")
def get_version(): 
    return "1.0.0"

async def main():
    await mcp.run_http_async(port=19091, host="0.0.0.0", transport="sse")
    
if __name__ == "__main__":
    asyncio.run(main())
    