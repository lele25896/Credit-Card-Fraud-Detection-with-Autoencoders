terraform {
  required_providers {
    google = { source = "hashicorp/google", version = "~> 5.0" }
  }

  # State remoto con locking. Bucket creato una-tantum (vedi BACKEND-SETUP.md).
  backend "gcs" {
    bucket = "fraud-detector-499714-tfstate"
    prefix = "cloud-run"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
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
  repository_id = var.service_name
  format        = "DOCKER"
  location      = var.region
  depends_on    = [google_project_service.artifactregistry]
}

resource "google_cloud_run_v2_service" "fraud_api" {
  name       = var.service_name
  location   = var.region
  depends_on = [google_project_service.run]

  template {
    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.service_name}/${var.service_name}:latest"
      resources { limits = { memory = "2Gi" } }
      liveness_probe {
        http_get { path = "/health" }
      }
    }
  }

  # ponytail: CI/CD owns revisions — ignore template + gcloud-stamped client metadata
  lifecycle {
    ignore_changes = [template, client, client_version]
  }
}

resource "google_cloud_run_v2_service_iam_member" "public" {
  project  = google_cloud_run_v2_service.fraud_api.project
  location = google_cloud_run_v2_service.fraud_api.location
  name     = google_cloud_run_v2_service.fraud_api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
