"""A stand-in for complaintops.mcp_server that needs no model, database or embedding model.

Same three tool names and the same output shapes as the real server, with canned classifier and search answers, so the
agent's rules can be checked on a fresh machine (GitHub Actions) where the 67 MB model and the search index do not exist.
The review queue is the real one (SQLite), so what the agent writes to it is tested for real.
    python -m tests.stub_mcp_server       (the agent starts it with ComplaintAgent(server_module="tests.stub_mcp_server"))
"""
from mcp.server.mcpserver import MCPServer

from complaintops import tools

mcp = MCPServer("complaintops-stub")

SURE = {"label": "Debt collection | Attempts to collect debt not owed", "confidence": 0.91,
        "alternatives": [{"label": "Debt collection | Attempts to collect debt not owed", "confidence": 0.91},
                         {"label": "Debt collection | Communication tactics", "confidence": 0.05},
                         {"label": "Other", "confidence": 0.02}]}
UNSURE = {"label": "Other", "confidence": 0.52,
          "alternatives": [{"label": "Other", "confidence": 0.52},
                           {"label": "Checking or savings account | Managing an account", "confidence": 0.31},
                           {"label": "Checking or savings account | Problem with a lender or other company charging your account",
                            "confidence": 0.09}]}
NEIGHBOURS = ["Checking or savings account | Managing an account"] * 3 + ["Other", "Credit card | Fees or interest"]


@mcp.tool()
def classify_complaint(text: str) -> dict:
    """Canned classifier: a complaint that mentions a collector is confident, everything else is not."""
    r = SURE if "collector" in text.lower() else UNSURE
    product, _, issue = r["label"].partition(" | ")
    return {**r, "product": product, "issue": issue or "(no specific issue)"}


@mcp.tool()
def search_similar_complaints(text: str, k: int = 5, product: str | None = None) -> dict:
    """Canned search: five neighbours, three of which agree on one label."""
    return {"results": [{"id": i, "similarity": 0.9 - i / 20, "label": label, "product": label.split(" | ")[0],
                         "date": "2023-01-01", "snippet": f"similar complaint {i}"}
                        for i, label in enumerate(NEIGHBOURS[:k])]}


@mcp.tool()
def flag_for_review(text: str, reason: str, predicted_label: str | None = None, confidence: float | None = None,
                    summary: str | None = None, suggestions: list[dict] | None = None) -> dict:
    """The real review queue."""
    return tools.flag_for_review(text, reason, predicted_label, confidence, summary, suggestions)


if __name__ == "__main__":
    mcp.run()
