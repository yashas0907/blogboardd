"""Tests for graph wiring and tutorial/news agent dry-run paths."""

from blogboard.graph.graph import build_graph


def test_graph_builds_successfully():
    graph = build_graph()
    # All expected nodes present
    node_names = set(graph.get_graph().nodes.keys())
    assert {"tutorial_agent", "news_agent", "validator"} <= node_names


def test_tutorial_dry_run_via_graph():
    graph = build_graph()
    final = graph.invoke(
        {"dry_run": True},
        config={"configurable": {"thread_id": "test-tutorial"}},
    )
    assert final["revision_needed"] is False
    assert final["domain"] != "ainews"
    assert "content" in final


def test_ainews_dry_run_via_graph():
    graph = build_graph()
    final = graph.invoke(
        {"dry_run": True, "domain": "ainews"},
        config={"configurable": {"thread_id": "test-news"}},
    )
    assert final["domain"] == "ainews"
    assert final["revision_needed"] is False


def test_requested_domain_is_respected():
    """Regression: --domain nlp must generate for nlp, not auto-scheduled dl.

    The tutorial agent previously ignored a preset domain whenever no topic
    was set — auto-scheduling overrode the explicit request.
    """
    from blogboard.agents.tutorial_agent.agent import tutorial_node

    result = tutorial_node({"dry_run": True, "domain": "nlp"})
    assert result["domain"] == "nlp"
