param(
    [Parameter(Mandatory = $true)]
    [string]$OfferId,

    [string]$GitRepo = "https://github.com/achetverikov/demixing_model.git",
    [string]$GitRef = "main",
    [string]$Image = "andreychetverikov/demixing-vast:latest",
    [int]$Disk = 40,
    [string]$S3Prefix = $(if ($env:VAST_SMOKE_S3_PREFIX) { $env:VAST_SMOKE_S3_PREFIX } else { "demixing/vast_smoke" }),
    [string]$Registry = $(if ($env:VAST_SMOKE_REGISTRY) { $env:VAST_SMOKE_REGISTRY } else { "averaged_surfaces_vast_smoke" }),
    [int]$NSimulations = 100,
    [int]$NSamples = 20,
    [string]$GpuWorkers = "0",
    [string]$ExtraArgs = "--grid-level 1 --chunk-size 5 --max-chunks 1"
)

$ErrorActionPreference = "Stop"

function Import-EnvFile {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return
    }

    Get-Content $Path | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            return
        }
        $line = $line -replace '^export\s+', ''
        $parts = $line.Split("=", 2)
        if ($parts.Count -ne 2) {
            return
        }
        $name = $parts[0].Trim()
        $value = $parts[1].Trim().Trim('"').Trim("'")
        if ($name) {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

Import-EnvFile ".env.docker"
Import-EnvFile ".env"

$required = @(
    "REDIS_HOST",
    "REDIS_PASSWORD",
    "S3_BUCKET",
    "S3_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY"
)

$missing = @()
foreach ($name in $required) {
    if (-not [Environment]::GetEnvironmentVariable($name, "Process")) {
        $missing += $name
    }
}

if ($missing.Count -gt 0) {
    throw "Missing required env vars: $($missing -join ', ')"
}

$vastKey = $env:VASTAPIKEY
if (-not $vastKey) {
    $vastKey = $env:VAST_API_KEY
}
if ($vastKey) {
    vastai set api-key $vastKey
}

$redisPort = if ($env:REDIS_PORT) { $env:REDIS_PORT } else { "6379" }
$redisUser = if ($env:REDIS_USERNAME) { $env:REDIS_USERNAME } else { "default" }
$awsRegion = if ($env:AWS_DEFAULT_REGION) { $env:AWS_DEFAULT_REGION } else { "auto" }
$avgDir = "results/$Registry"
$bundleDir = "results/${Registry}_bundles"

$vastEnv = @(
    "-e GIT_REPO=$GitRepo",
    "-e GIT_REF=$GitRef",
    "-e REPO_DIR=/workspace/demixing_model",
    "-e REDIS_HOST=$env:REDIS_HOST",
    "-e REDIS_PORT=$redisPort",
    "-e REDIS_USERNAME=$redisUser",
    "-e REDIS_PASSWORD=$env:REDIS_PASSWORD",
    "-e S3_BUCKET=$env:S3_BUCKET",
    "-e S3_PREFIX=$S3Prefix",
    "-e S3_ENDPOINT_URL=$env:S3_ENDPOINT_URL",
    "-e AWS_ACCESS_KEY_ID=$env:AWS_ACCESS_KEY_ID",
    "-e AWS_SECRET_ACCESS_KEY=$env:AWS_SECRET_ACCESS_KEY",
    "-e AWS_DEFAULT_REGION=$awsRegion",
    "-e AVERAGED_SURFACES_DIR=$avgDir",
    "-e COMPLETION_REGISTRY=$Registry",
    "-e BUNDLE_OUTPUT=1",
    "-e BUNDLE_DIR=$bundleDir",
    "-e N_SIMULATIONS=$NSimulations",
    "-e N_SAMPLES=$NSamples",
    "-e GPU_WORKERS=$GpuWorkers",
    "-e EXTRA_ARGS=`"$ExtraArgs`""
) -join " "

$onStart = 'cd /workspace && git clone "$GIT_REPO" "$REPO_DIR" && cd "$REPO_DIR" && git checkout "$GIT_REF" && bash cloud/vast_worker.sh'

vastai create instance $OfferId `
    --image $Image `
    --disk $Disk `
    --ssh `
    --direct `
    --env $vastEnv `
    --onstart-cmd $onStart
