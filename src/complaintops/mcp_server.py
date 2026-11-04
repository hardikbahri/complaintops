"""MCP server exposing the three complaint tools. Milestone 2, step 3.

Any MCP client can connect: our LangGraph agent, Claude Desktop, an IDE. Run over stdio (the client launches it):
    python -m complaintops.mcp_server
The tool docstrings below are what the LLM reads when it decides which tool to call, so they say when to use each.
"""
from mcp.server.mcpserver import MCPServer

from complaintops import tools

mcp = MCPServer(
    "complaintops",
    instructions="Tools for triaging consumer financial complaints: classify, find similar past complaints, "
                 "and escalate to a human reviewer.")


@mcp.tool()
def classify_complaint(text: str) -> dict:
    """Classify a consumer complaint into a product and an issue using the fine-tuned classifier.

    Returns the label, product, issue, a confidence between 0 and 1 and the runner-up classes. Always call
    this first. A confidence below about 0.6 means the model is unsure.
    """
    return tools.classify_complaint(text)


@mcp.tool()
def search_similar_complaints(text: str, k: int = 5, product: str | None = None) -> dict:
    """Find past complaints most similar to this text, with how each one was labelled.

    Use it when the classifier is unsure, to see how comparable complaints were handled. `k` is how many to
    return (1 to 10); `product` optionally restricts the search to one product.
    """
    return tools.search_similar_complaints(text, k, product)


@mcp.tool()
def flag_for_review(text: str, reason: str, predicted_label: str | None = None,
                    confidence: float | None = None) -> dict:
    """Send a complaint to the human review queue when it cannot be labelled reliably.

    Give a short `reason` (for example "low confidence and the similar complaints disagree"). Returns the case
    id and the current queue length.
    """
    return tools.flag_for_review(text, reason, predicted_label, confidence)


def main():
    mcp.run()  # stdio


if __name__ == "__main__":
    main()
