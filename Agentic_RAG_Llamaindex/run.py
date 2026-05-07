import asyncio
from llama_index.core.workflow import Context

from agent import alfred


async def ask(ctx: Context, question: str):
    response = await alfred.run(question, ctx=ctx)

    print("\nUSER:")
    print(question)

    print("\nALFRED:")
    print(str(response))


async def main():
    ctx = Context(alfred)

    await ask(ctx, "Tell me about Lady Ada Lovelace and give me her email.")
    await ask(ctx, "今晚巴黎适合放烟花吗？")
    await ask(ctx, "我要和 Dr. Nikola Tesla 聊无线能量传输，帮我准备一些话题。")
    await ask(ctx, "刚才那位嘉宾还有什么适合闲聊的话题？")


if __name__ == "__main__":
    asyncio.run(main())