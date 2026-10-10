"""Provider-agnostic LLM callable. Whichever key is present gets used."""
import os


def make_llm(model: str | None = None, temperature: float = 0):
    """Returns a plain callable: prompt string in, text out."""
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        chat = ChatOpenAI(model=model or os.environ.get("LLM_MODEL", "gpt-4o"),
                          temperature=temperature)
    elif os.environ.get("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        chat = ChatAnthropic(
            model=model or os.environ.get("LLM_MODEL", "claude-sonnet-4-5-20250929"),
            temperature=temperature)
    else:
        raise RuntimeError(
            "No model key found. Put OPENAI_API_KEY or ANTHROPIC_API_KEY in .env")

    def call(prompt: str) -> str:
        out = chat.invoke(prompt).content
        if isinstance(out, list):            # some providers return content blocks
            out = "".join(b.get("text", "") for b in out if isinstance(b, dict))
        return out

    return call
