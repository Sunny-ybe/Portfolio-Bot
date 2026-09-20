from anthropic import Anthropic

from app.config import settings
from app.models import ChatResponse, Citation

_client = Anthropic(api_key=settings.anthropic_api_key)

UNGROUNDED_ANSWER = "I don't have evidence to support an answer to that."

SYSTEM_PROMPT = """You answer questions about the site owner using ONLY the evidence chunks provided.
Every claim in your answer must be traceable to a chunk_id you list in citations.
If the evidence doesn't support an answer, explain that in "answer" and leave "citations" empty.
Never state anything not directly supported by the provided evidence chunks."""

ANSWER_TOOL = {
    "name": "submit_answer",
    "description": "Submit the answer to the user's question, grounded in the provided evidence chunks.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The answer to the question, or an explanation that no evidence supports an answer.",
            },
            "citations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "chunk_id values from the evidence block that support the answer.",
            },
        },
        "required": ["answer", "citations"],
    },
}


def answer_from_chunks(question: str, matches: list[tuple[dict, float]]) -> ChatResponse:
    chunk_by_id = {chunk["id"]: chunk for chunk, _ in matches}
    evidence_block = "\n\n".join(
        f"[{chunk['id']}] (source: {chunk['source']}, tier: {chunk['evidence_tier']})\n{chunk['text']}"
        for chunk, _ in matches
    )

    response = _client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=[ANSWER_TOOL],
        tool_choice={"type": "tool", "name": "submit_answer"},
        messages=[
            {
                "role": "user",
                "content": f"Evidence:\n{evidence_block}\n\nQuestion: {question}",
            }
        ],
    )

    tool_call = next(block for block in response.content if block.type == "tool_use")
    parsed = tool_call.input

    valid_citations = [
        Citation(
            chunk_id=cid,
            source=chunk_by_id[cid]["source"],
            text=chunk_by_id[cid]["text"],
            evidence_tier=chunk_by_id[cid]["evidence_tier"],
        )
        for cid in parsed.get("citations", [])
        if cid in chunk_by_id
    ]

    if not valid_citations:
        return ChatResponse(answer=UNGROUNDED_ANSWER, citations=[], grounded=False)

    return ChatResponse(answer=parsed["answer"], citations=valid_citations, grounded=True)
