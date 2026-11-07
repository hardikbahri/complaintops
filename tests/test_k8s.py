"""The Kubernetes manifests in deploy/k8s, checked without a cluster (so this runs in CI).

A manifest that points a probe at a path the app does not serve, or a Service at the wrong port, only fails when it is
applied. These tests catch that early by comparing the YAML with the app and the Dockerfile.
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
K8S = ROOT / "deploy" / "k8s"


def load():
    docs = list(yaml.safe_load_all((K8S / "complaintops.yaml").read_text(encoding="utf-8")))
    return {d["kind"]: d for d in docs}


def container():
    return load()["Deployment"]["spec"]["template"]["spec"]["containers"][0]


def to_mib(q):
    return int(q[:-2]) if q.endswith("Mi") else int(q[:-2]) * 1024


def test_the_probes_point_at_routes_the_app_serves():
    from complaintops.api import app

    served = {r.path for r in app.routes}
    c = container()
    for probe in ("startupProbe", "readinessProbe", "livenessProbe"):
        assert c[probe]["httpGet"]["path"] in served, probe
        assert c[probe]["httpGet"]["port"] == c["ports"][0]["containerPort"]


def test_the_container_port_matches_the_dockerfile_and_the_service_reaches_it():
    dockerfile = (ROOT / "deploy" / "Dockerfile").read_text(encoding="utf-8")
    port = container()["ports"][0]["containerPort"]
    assert f"EXPOSE {port}" in dockerfile and f'"--port", "{port}"' in dockerfile
    svc = load()["Service"]
    assert svc["spec"]["ports"][0]["targetPort"] == port
    assert svc["spec"]["selector"] == load()["Deployment"]["spec"]["selector"]["matchLabels"]


def test_the_memory_limit_leaves_room_over_what_the_container_measured():
    mem = container()["resources"]
    assert to_mib(mem["requests"]["memory"]) >= 580  # measured: about 580 MB in use
    assert to_mib(mem["limits"]["memory"]) > to_mib(mem["requests"]["memory"])


def test_the_container_runs_as_a_non_root_user_without_extra_privileges():
    sc = container()["securityContext"]
    assert sc["runAsNonRoot"] is True and sc["runAsUser"] == 1000 and sc["allowPrivilegeEscalation"] is False
    assert "ALL" in sc["capabilities"]["drop"]


def test_no_secret_value_is_written_in_the_manifest_and_the_llm_key_is_optional():
    text = (K8S / "complaintops.yaml").read_text(encoding="utf-8")
    for needle in ("gsk_", "hf_", "sk-", "LANGFUSE"):
        assert needle not in text, needle  # tracing stays off, so visitors' text is never sent to a tracing service
    env = {e["name"]: e for e in container()["env"]}
    assert env["LLM_PROVIDER"]["value"] == "mock"
    assert env["GROQ_API_KEY"]["valueFrom"]["secretKeyRef"]["optional"] is True


def test_the_kind_cluster_publishes_the_services_node_port():
    cluster = yaml.safe_load((K8S / "kind-cluster.yaml").read_text(encoding="utf-8"))
    mapping = cluster["nodes"][0]["extraPortMappings"][0]
    assert mapping["containerPort"] == load()["Service"]["spec"]["ports"][0]["nodePort"]
