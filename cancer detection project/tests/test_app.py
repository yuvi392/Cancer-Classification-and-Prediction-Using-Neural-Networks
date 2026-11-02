from app import app

def test_pages_and_invalid_form():
    app.config["TESTING"] = True
    client = app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/about").status_code == 200
    assert client.get("/dashboard").status_code == 200
    response = client.post("/predict", data={})
    assert response.status_code == 400
