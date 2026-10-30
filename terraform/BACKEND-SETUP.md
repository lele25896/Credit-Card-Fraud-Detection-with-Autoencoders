# Remote GCS backend — one-time setup

The Terraform state moves from local to a GCS bucket (with versioning = locking + history).
Run once, locally, with `gcloud auth login` already done.

## 1. Create the state bucket

```powershell
gsutil mb -p fraud-detector-499714 -l europe-west1 gs://fraud-detector-499714-tfstate
gsutil versioning set on gs://fraud-detector-499714-tfstate
```

## 2. Migrate the local state into the bucket

From the `terraform/` folder:

```powershell
terraform init -migrate-state
# answer "yes" to the prompt about copying the existing state into the gcs backend
```

Then verify nothing changes:

```powershell
terraform plan   # expected: No changes. Infrastructure is up-to-date.
```

## 3. Permissions for apply in CI

The CI's SA (`github-ci@fraud-detector-499714.iam.gserviceaccount.com`) now also runs
`terraform apply`. It needs to be able to read/write the state and manage the infra:

```powershell
# access to the state bucket
gsutil iam ch serviceAccount:github-ci@fraud-detector-499714.iam.gserviceaccount.com:roles/storage.objectAdmin gs://fraud-detector-499714-tfstate

# infra management (if not already present)
gcloud projects add-iam-policy-binding fraud-detector-499714 --member=serviceAccount:github-ci@fraud-detector-499714.iam.gserviceaccount.com --role=roles/run.admin
gcloud projects add-iam-policy-binding fraud-detector-499714 --member=serviceAccount:github-ci@fraud-detector-499714.iam.gserviceaccount.com --role=roles/artifactregistry.admin
# managing google_project_service resources (list/enable API) in main.tf
gcloud projects add-iam-policy-binding fraud-detector-499714 --member=serviceAccount:github-ci@fraud-detector-499714.iam.gserviceaccount.com --role=roles/serviceusage.serviceUsageAdmin
```

> One-time prerequisite: the APIs must be enabled by hand the first time
> (`gcloud services enable cloudresourcemanager.googleapis.com run.googleapis.com artifactregistry.googleapis.com`),
> otherwise the first `terraform plan` fails with `SERVICE_DISABLED`.

## Notes

- The state is NO LONGER committed: `.gitignore` already excludes `*.tfstate`.
- `terraform.tfvars` is committed on purpose (no secrets, just project/region/name).
- On a PR, CI only runs `plan` (review); on push to `main` it runs `apply` then the deploy.
