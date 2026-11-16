import json
import asyncio

from mcp import ClientSession
from openai import AsyncOpenAI
from uvicorn import Server, Config
from fastapi import FastAPI
from mcp.client.sse import sse_client
from sse_starlette.sse import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware

openai = AsyncOpenAI(
    api_key="api-key",
    base_url="http://localhost:8888/v1",
    timeout=120,
)


class MCPServer:
    async def init_mcp_server(self, url):
        print(f"加载MCP Server: {url}")
        self.url = url
        self.client = sse_client(url)
        self.streams = await self.client.__aenter__()
        self.session = ClientSession(*self.streams)
        self.context = await self.session.__aenter__()
        await self.context.initialize()
        tools = await self.context.list_tools()
        prompts = await self.context.list_prompts()
        resources = await self.context.list_resources()
        print(f"tool: {[tool.name for tool in tools.tools]}")
        print(f"prompt: {[prompt.name for prompt in prompts.prompts]}")
        print(f"resource: {[resource.name for resource in resources.resources]}")

    async def close_mcp_server(self):
        print(f"关闭MCP Server: {self.url}")
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
        except Exception as e:
            print(f"关闭MCP Server异常: {e}")
        try:
            if self.client:
                await self.client.__aexit__(None, None, None)
        except Exception as e:
            print(f"关闭MCP Server异常: {e}")


class MCPClient:
    async def init_mcp_server(self, urls=[]):
        self.tools = []
        self.servers = []
        self.mapping = {}
        for url in urls:
            mcpServer = MCPServer()
            try:
                await mcpServer.init_mcp_server(url)
                for tool in (await mcpServer.context.list_tools()).tools:
                    self.tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.input_schema,
                            },
                        }
                    )
                    self.mapping[tool.name] = mcpServer
                self.servers.append(mcpServer)
            except Exception as e:
                print(f"加载MCP Server异常: {e}")
                await mcpServer.close_mcp_server()
        print(f"加载MCP Server完成: {len(self.servers)}")

    async def close_mcp_server(self):
        for server in self.servers:
            await server.close_mcp_server()

    def tool_fill(self, tools_params, tool):
        if tool.index not in tools_params:
            tools_params[tool.index] = tool
        if tool.id:
            tools_params[tool.index].id = tool.id
        if tool.function:
            if tools_params[tool.index].function is None:
                tools_params[tool.index].function = tool.function
            if tool.function.name:
                tools_params[tool.index].function.name = tool.function.name
            if tool.function.arguments:
                if tools_params[tool.index].function.arguments is None:
                    tools_params[tool.index].function.arguments = ""
                tools_params[tool.index].function.arguments += tool.function.arguments

    def tool_text(self, content):
        if content.type == "text":
            return content.text
        print(f"不支持的内容类型: {content}")
        return ""

    async def call_tool(self, tools_params, messages):
        for _, tool in tools_params.items():
            if (
                not tool.id
                and not tool.type
                and not tool.function
                and not tool.function.name
            ):
                print(f"调用工具参数错误: {tool}")
                continue
            if tool.function.name not in self.mapping:
                print(f"调用工具没有映射: {tool}")
                continue
            try:
                arguments = (
                    json.loads(tool.function.arguments)
                    if tool.function.arguments
                    else {}
                )
                response = await asyncio.wait_for(
                    self.mapping[tool.function.name].context.call_tool(
                        tool.function.name,
                        arguments,
                    ),
                    timeout=120,
                )
                content = "".join(
                    list(map(lambda v: self.tool_text(v), response.content or []))
                )
                print(f"调用工具: {tool} - {content}")
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": tool.id,
                                    "type": tool.type,
                                    "function": {
                                        "name": tool.function.name,
                                        "arguments": tool.function.arguments or "{}",
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": tool.id,
                            "content": content,
                        },
                    ]
                )
            except Exception as e:
                print(f"调用工具异常: {tool} - {e}")
        return messages

    async def chat(self, model, stream, messages, tool_call_count=0):
        params = {
            "model": model,
            "tools": self.tools,
            "stream": stream,
            "messages": messages,
        }
        print(f"会话请求: {params}")
        text = ""
        response = None
        tools_params = {}
        try:
            response = await openai.chat.completions.create(**params)
            if stream:
                async for chunk in response:
                    if chunk and chunk.choices:
                        print(f"会话响应: {chunk}")
                        for choice in chunk.choices:
                            delta = choice.delta
                            reason = choice.finish_reason
                            if delta.content:
                                text += delta.content
                                yield delta.content
                            if delta.tool_calls:
                                for tool in delta.tool_calls:
                                    self.tool_fill(tools_params, tool)
                            if reason:
                                if reason == "stop":
                                    print(f"会话完成: {reason} = {text}")
                                    messages.append(
                                        {
                                            "role": "assistant",
                                            "content": text,
                                        }
                                    )
                                elif reason == "tool_calls":
                                    tool_call_count += 1
                                    if tool_call_count > 8:
                                        message = "工具调用次数超限"
                                        messages.append(
                                            {"role": "assistant", "content": message}
                                        )
                                        yield message
                                        return
                                    messages = await self.call_tool(
                                        tools_params, messages
                                    )
                                    async for recursion_chunk in self.chat(
                                        model, stream, messages, tool_call_count
                                    ):
                                        yield recursion_chunk
                                    break
                                else:
                                    print(f"其他结束原因: {reason}")
            else:
                for choice in response.choices:
                    print(f"会话响应: {choice}")
                    message = choice.message
                    reason = choice.finish_reason
                    if message.content:
                        text += message.content
                    if message.tool_calls:
                        for tool in message.tool_calls:
                            tools_params[tool.index] = tool
                    if reason == "stop":
                        messages.append(
                            {
                                "role": "assistant",
                                "content": text,
                            }
                        )
                    elif reason == "tool_calls":
                        tool_call_count += 1
                        if tool_call_count > 8:
                            message = "工具调用次数超限"
                            messages.append({"role": "assistant", "content": message})
                            yield message
                            return
                        messages = await self.call_tool(tools_params, messages)
                        async for recursion_chunk in self.chat(
                            model, stream, messages, tool_call_count
                        ):
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
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.get("/chat")
async def chat(model: str, message: str, stream: bool = True):
    messages = [
        {
            "role": "system",
            "content": "你是一个中文AI助手",
        },
        {
            "role": "user",
            "content": message,
        },
    ]
    if stream:
        return EventSourceResponse(mcpClient.chat(model, stream, messages))
    else:
        content = ""
        async for value in mcpClient.chat(model, stream, messages):
            content += value
        return content


async def main():
    await mcpClient.init_mcp_server(["http://localhost:19090/sse"])
    config = Config(app, host="0.0.0.0", port=18080)
    server = Server(config)
    try:
        await server.serve()
    finally:
        await mcpClient.close_mcp_server()


if __name__ == "__main__":
    asyncio.run(main())

# 查询广州天气
# 今天广州出门搭乘什么交通工具
