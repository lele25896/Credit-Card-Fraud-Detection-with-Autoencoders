terraform {
  required_providers {
    google = { source = "hashicorp/google", version = "~> 5.0" }
  }
}

provider "google" {
  project = "fraud-detector-499714"
  region  = "europe-west1"
}

resource "google_project_service" "run" {
  service            = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "artifactregistry" {
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "fraud_api" {
  repository_id = "fraud-api"
  format        = "DOCKER"
  location      = "europe-west1"
  depends_on    = [google_project_service.artifactregistry]
}

resource "google_cloud_run_v2_service" "fraud_api" {
  name       = "fraud-api"
  location   = "europe-west1"
  depends_on = [google_project_service.run]

  template {
    containers {
      image = "europe-west1-docker.pkg.dev/fraud-detector-499714/fraud-api/fraud-api:latest"
      resources { limits = { memory = "1Gi" } }
      liveness_probe {
        http_get { path = "/health" }
      }
    }
  }

  # ponytail: CI/CD owns revisions — ignore template drift after first apply
  lifecycle {
    ignore_changes = [template]
  }
}

resource "google_cloud_run_v2_service_iam_member" "public" {
  project  = google_cloud_run_v2_service.fraud_api.project
  location = google_cloud_run_v2_service.fraud_api.location
  name     = google_cloud_run_v2_service.fraud_api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
