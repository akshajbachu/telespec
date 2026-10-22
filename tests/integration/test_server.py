import io

from fastapi.testclient import TestClient

from telespec.server.app import ANALYSES, create_app


def test_health_demo_export_and_upload(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    assert client.get("/api/health").json()["local_only"]
    demo = client.post("/api/demo")
    assert demo.status_code == 200
    analysis_id = demo.json()["id"]
    assert client.get(f"/api/export/{analysis_id}").status_code == 200
    markdown = client.get(f"/api/export/{analysis_id}?format=markdown")
    assert markdown.status_code == 200 and "# telespec report" in markdown.text
    csv = "timestamp,latency\n" + "\n".join(f"{i * 60},{10 + i % 5}" for i in range(80))
    uploaded = client.post(
        "/api/analyze",
        files={"file": ("metric.csv", io.BytesIO(csv.encode()), "text/csv")},
        data={"column": "latency", "bootstraps": "10"},
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["filename"] == "metric.csv"
    assert client.get("/api/export/missing").status_code == 404
    ANALYSES.clear()


def test_upload_validation(tmp_path):
    client = TestClient(create_app(tmp_path))
    response = client.post(
        "/api/analyze",
        files={"file": ("bad.csv", b"nope", "text/csv")},
        data={"bootstraps": "10"},
    )
    assert response.status_code == 422
    response = client.post(
        "/api/analyze",
        files={"file": ("x.csv", b"timestamp,value\n0,1\n1,2\n2,3", "text/csv")},
        data={"train_fraction": "0.99"},
    )
    assert response.status_code == 422
