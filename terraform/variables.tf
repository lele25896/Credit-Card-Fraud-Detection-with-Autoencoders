variable "project_id" {
  type        = string
  description = "GCP project ID"
}

variable "region" {
  type        = string
  description = "GCP region for Cloud Run + Artifact Registry"
  default     = "europe-west1"
}

variable "service_name" {
  type        = string
  description = "Name reused for Cloud Run service and Artifact Registry repo"
  default     = "fraud-api"
}

# ponytail: no module + no per-env workspaces — single project, one env.
# Add a module and dev/prod tfvars only if a second environment appears.

# test CI: plan-on-PR
