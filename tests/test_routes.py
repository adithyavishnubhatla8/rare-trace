import pytest
from app import app
from analysis.run_analysis import execute_full_pipeline

@pytest.fixture(scope="module")
def client():
    app.config['TESTING'] = True
    # Ensure database is initialized before testing routes
    execute_full_pipeline()
    with app.test_client() as client:
        yield client

def test_dashboard_route(client):
    response = client.get('/dashboard')
    assert response.status_code == 200
    assert b"Clinical Analytics Dashboard" in response.data

def test_patients_route(client):
    response = client.get('/patients')
    assert response.status_code == 200
    assert b"Patient Cohort Records" in response.data

def test_anomalies_route(client):
    response = client.get('/anomalies')
    assert response.status_code == 200
    assert b"Candidate Rare-Disease Cases" in response.data

def test_model_analysis_route(client):
    response = client.get('/model-analysis')
    assert response.status_code == 200
    assert b"DBSCAN Model Analysis" in response.data

def test_api_statistics(client):
    response = client.get('/api/statistics')
    assert response.status_code == 200
    json_data = response.get_json()
    assert "total_patients" in json_data
    assert "candidate_rare_cases" in json_data
