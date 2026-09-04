param(
    [ValidateSet("all","tinystories","wikipedia","cosmopedia","fineweb-edu","openwebmath","ultrachat")]
    [string]$Dataset = "all",
    [string]$Org = "MlBricks",
    [long]$TargetTokens = 1000000000,
    [switch]$DryRun,
    [switch]$KeepBuild
)

$ErrorActionPreference = "Stop"

Write-Host "Installing/updating Hugging Face publishing dependencies..." -ForegroundColor Cyan
python -m pip install -U huggingface_hub datasets transformers pyarrow
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $env:HF_TOKEN) {
    Write-Host "" 
    Write-Host "HF_TOKEN is not set." -ForegroundColor Yellow
    Write-Host "Use 'hf auth login' first, or set HF_TOKEN as a secret/environment variable." -ForegroundColor Yellow
    Write-Host "Do not paste the token into source code." -ForegroundColor Yellow
}

$argsList = @(
    "tools/publish_mlbricks_datasets.py",
    "--dataset", $Dataset,
    "--org", $Org,
    "--target-tokens", $TargetTokens
)
if ($DryRun) { $argsList += "--dry-run" }
if ($KeepBuild) { $argsList += "--keep-build" }

python @argsList
exit $LASTEXITCODE
