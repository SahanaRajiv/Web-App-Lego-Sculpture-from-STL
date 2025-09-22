param(
  [Parameter(Mandatory=$true)][string]$StlPath,
  [string]$Url = "http://127.0.0.1:5000/upload"
)

if (-not (Test-Path $StlPath)) {
  Write-Error "File not found: $StlPath"
  exit 1
}

curl -Method Post -Uri $Url -Form @{
  file = Get-Item $StlPath
  grid_voxel_count = 50
  grid_direction = 'z'
  generate_pdf = ''
  color_by_shape = ''
  remove_hanging_bricks = ''
}

Write-Host "Submitted. Open http://127.0.0.1:5000/history to view results."