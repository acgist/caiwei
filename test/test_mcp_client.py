import json
import asyncio

from mcp import ClientSession
from openai import AsyncOpenAI
from uvicorn import Server, Config
from fastapi import FastAPI, Request
from mcp.client.sse import sse_client
from sse_starlette.sse import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware

openai = AsyncOpenAI(api_key = "api-key", base_url = "https://api.deepseek.com")

class MCPServer:
    async def init_mcp_server(self, url):
        print(f"加载MCP Server: {url}")
        self.url     = url
        self.client  = sse_client(url)
        self.streams = await self.client.__aenter__()
        self.session = ClientSession(*self.streams)
        self.context = await self.session.__aenter__()
        await self.context.initialize()
        tools     = await self.context.list_tools()
        prompts   = await self.context.list_prompts()
        resources = await self.context.list_resources()
        print(f"tool: {[tool.name for tool in tools.tools]}")
        print(f"prompt: {[prompt.name for prompt in prompts.prompts]}")
        print(f"resource: {[resource.name for resource in resources.resources]}")

    async def close_mcp_server(self):
        print(f"关闭MCP Server: {self.url}")
        if self.session:
            await self.session.__aexit__(None, None, None)
        if self.client:
            await self.client.__aexit__(None, None, None)

class MCPClient:
    async def init_mcp_server(self, urls = []):
        self.tools   = []
        self.servers = []
        self.mapping = {}
        for url in urls:
            mcpServer = MCPServer()
            await mcpServer.init_mcp_server(url)
            for tool in (await mcpServer.context.list_tools()).tools:
                self.tools.append({
                    "type"    : "function",
                    "function": {
                        "name"       : tool.name,
                        "parameters" : tool.input_schema,
                        "description": tool.description,
                    }})
                self.mapping[tool.name] = mcpServer
            self.servers.append(mcpServer)
        print(f"加载MCP Server完成: {len(self.servers)}")

    async def close_mcp_server(self):
        for server in self.servers:
            try:
                await server.close_mcp_server()
            except Exception as e:
                print(f"关闭MCP Server异常: {e}")

    async def call_tool(self, tools_params, messages):
        for _, tool in tools_params.items():
            if not tool.function:
                print(f"调用工具参数错误: {tool}")
                continue
            if not tool.function.name:
                print(f"调用工具参数错误: {tool}")
                continue
            if tool.function.name not in self.mapping:
                print(f"调用工具没有映射: {tool}")
                continue
            try:
                params   = json.loads(tool.function.arguments) if tool.function.arguments else "{}"
                response = await self.mapping[tool.function.name].context.call_tool(tool.function.name, params)
                content  = "".join(list(map(lambda v: v.text, response.content)))
                print(f"调用工具: {tool} - {content}")
                messages.extend([
                    {
                        "role"      : "assistant",
                        "content"   : None,
                        "tool_calls": [tool]
                    },
                    {
                        "role"        : "tool",
                        "content"     : content,
                        "tool_call_id": tool.id,
                    }
                ])
            except Exception as e:
                print(f"调用工具异常: {tool} - {e}")
        return messages

    async def chat(self, stream, messages):
        params = {
            "model"   : "deepseek-chat",
            "tools"   : self.tools,
            "stream"  : stream,
            "messages": messages,
        }
        print(f"会话请求: {params}")
        text         = ""
        response     = None
        tools_params = {}
        try:
            response = await openai.chat.completions.create(**params)
            if stream:
                async for chunk in response:
                    if chunk and chunk.choices:
                        print(f"会话响应: {chunk}")
                        for choice in chunk.choices:
                            delta  = choice.delta
                            reason = choice.finish_reason
                            if delta.content:
                                text += delta.content
                                yield delta.content
                            if delta.tool_calls:
                                for tool in delta.tool_calls:
                                    if tool.index not in tools_params:
                                        tools_params[tool.index] = tool
                                    if tool.id:
                                        tools_params[tool.index].id = tool.id
                                    if tool.function:
                                        if tool.function.name:
                                            tools_params[tool.index].function.name = tool.function.name
                                        if tool.function.arguments:
                                            tools_params[tool.index].function.arguments += tool.function.arguments
                            if reason:
                                print(f"会话完成: {reason} = {text}")
                                if reason == "stop":
                                    messages.append({
                                        "role"   : "assistant",
                                        "content": text,
                                    })
                                elif reason == "tool_calls":
                                    messages = await self.call_tool(tools_params, messages)
                                    async for recursion_chunk in self.chat(stream, messages):
                                        yield recursion_chunk
                                    break
                                else:
                                    print(f"其他结束原因: {reason}")
            else:
                for choice in response.choices:
                    print(f"会话响应: {choice}")
                    message = choice.message
                    reason  = choice.finish_reason
                    if message.content:
                        text += message.content
                    if message.tool_calls:
                        for tool in message.tool_calls:
                            tools_params[tool.index] = tool
                    if reason == "stop":
                        messages.append({
                            "role"   : "assistant",
                            "content": text,
                        })
                    elif reason == "tool_calls":
                        messages = await self.call_tool(tools_params, messages)
                        async for recursion_chunk in self.chat(stream, messages):
                            text += recursion_chunk
                    else:
                        print(f"其他结束原因: {reason}")
                if text:
                    yield text
        except Exception as e:
            print(f"会话异常: {e}")
        finally:
            if stream and response:
                await response.close()

mcpClient = MCPClient()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
    allow_credentials = True,
)

@app.get("/chat")
async def chat(message: str, stream: bool = True):
    messages = [
        {
            "role"   : "system",
            "content": "你是一个中文AI助手！"
        },
        {
            "role"   : "user",
            "content": message
        }
    ]
    return EventSourceResponse(mcpClient.chat(stream, messages))

async def main():
    await mcpClient.init_mcp_server(["http://localhost:19091/sse"])
    config = Config(app, host = "0.0.0.0", port = 18081)
    server = Server(config)
    await server.serve()
    await mcpClient.close_mcp_server()

if __name__ == "__main__":
    asyncio.run(main())
