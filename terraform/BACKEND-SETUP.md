# Backend remoto GCS — setup una-tantum

Lo state Terraform passa da locale a bucket GCS (con versioning = locking + storia).
Da eseguire una volta sola, in locale con `gcloud auth login` fatto.

## 1. Crea il bucket di state

```powershell
gsutil mb -p fraud-detector-499714 -l europe-west1 gs://fraud-detector-499714-tfstate
gsutil versioning set on gs://fraud-detector-499714-tfstate
```

## 2. Migra lo state locale nel bucket

Dalla cartella `terraform/`:

```powershell
terraform init -migrate-state
# rispondi "yes" alla domanda di copia dello state esistente nel backend gcs
```

Poi verifica che nulla cambi:

```powershell
terraform plan   # atteso: No changes. Infrastructure is up-to-date.
```

## 3. Permessi per l'apply in CI

Il SA della CI (`github-ci@fraud-detector-499714.iam.gserviceaccount.com`) ora fa
anche `terraform apply`. Serve che possa leggere/scrivere lo state e gestire l'infra:

```powershell
# accesso allo state bucket
gsutil iam ch serviceAccount:github-ci@fraud-detector-499714.iam.gserviceaccount.com:roles/storage.objectAdmin gs://fraud-detector-499714-tfstate

# gestione infra (se non già presenti)
gcloud projects add-iam-policy-binding fraud-detector-499714 --member=serviceAccount:github-ci@fraud-detector-499714.iam.gserviceaccount.com --role=roles/run.admin
gcloud projects add-iam-policy-binding fraud-detector-499714 --member=serviceAccount:github-ci@fraud-detector-499714.iam.gserviceaccount.com --role=roles/artifactregistry.admin
# gestione delle google_project_service (list/enable API) nel main.tf
gcloud projects add-iam-policy-binding fraud-detector-499714 --member=serviceAccount:github-ci@fraud-detector-499714.iam.gserviceaccount.com --role=roles/serviceusage.serviceUsageAdmin
```

> Prerequisito una-tantum: le API vanno abilitate a mano la prima volta
> (`gcloud services enable cloudresourcemanager.googleapis.com run.googleapis.com artifactregistry.googleapis.com`),
> altrimenti il primo `terraform plan` fallisce con `SERVICE_DISABLED`.

## Note

- Lo state NON è più committato: `.gitignore` già esclude `*.tfstate`.
- `terraform.tfvars` è committato apposta (nessun segreto, solo project/region/nome).
- Su PR la CI fa solo `plan` (review); su push a `main` fa `apply` poi il deploy.
