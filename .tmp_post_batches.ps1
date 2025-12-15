$ErrorActionPreference = "Stop"
$apiUrl = "http://localhost:8000/batch/predict"
$batches = Get-ChildItem "data/drift_batches" -Directory | Select-Object -First 10
Write-Host "Found $($batches.Count) batches"
$results = @()
foreach ($b in $batches) {
  Write-Host "Posting $($b.Name)"
  $form = @{
    application           = Get-Item (Join-Path $b.FullName "application.csv")
    bureau                = Get-Item (Join-Path $b.FullName "bureau.csv")
    bureau_balance        = Get-Item (Join-Path $b.FullName "bureau_balance.csv")
    previous_application  = Get-Item (Join-Path $b.FullName "previous_application.csv")
    credit_card_balance   = Get-Item (Join-Path $b.FullName "credit_card_balance.csv")
    installments_payments = Get-Item (Join-Path $b.FullName "installments_payments.csv")
    pos_cash_balance      = Get-Item (Join-Path $b.FullName "POS_CASH_balance.csv")
  }
  try {
    $resp = Invoke-WebRequest -Uri $apiUrl -Method Post -ContentType "multipart/form-data" -Form $form
    $json = $resp.Content | ConvertFrom-Json
    $results += [pscustomobject]@{
      batch         = $b.Name
      success       = $json.success
      n_predictions = $json.n_predictions
      batch_id      = $json.batch_id
      timestamp     = $json.timestamp
    }
  }
  catch {
    $results += [pscustomobject]@{
      batch         = $b.Name
      success       = $false
      n_predictions = 0
      batch_id      = $null
      timestamp     = (Get-Date).ToString("s")
      error         = $_.Exception.Message
    }
    Write-Warning "Failed $($b.Name): $($_.Exception.Message)"
  }
}
$results | Format-Table -AutoSize
if (!(Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" | Out-Null }
$results | ConvertTo-Json -Depth 3 | Out-File -Encoding utf8 "logs/posted_drift_batches.json"
Write-Host "Saved summary to logs/posted_drift_batches.json"
