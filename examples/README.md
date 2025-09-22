# Examples

## Upload with cURL (bash)
```bash
curl -X POST http://127.0.0.1:5000/upload \
  -F "file=@/absolute/path/to/your_model.stl" \
  -F "grid_voxel_count=50" \
  -F "grid_direction=z" \
  -F "generate_pdf=" \
  -F "color_by_shape=" \
  -F "remove_hanging_bricks="
```

## Upload with PowerShell
```powershell
$uri = "http://127.0.0.1:5000/upload"
$stl = "D:\\Projects\\lego_stl\\uploads\\StanfordBunny_fixed.stl"

curl -Method Post -Uri $uri -Form @{
  file = Get-Item $stl
  grid_voxel_count = 50
  grid_direction = 'z'
  generate_pdf = ''
  color_by_shape = ''
  remove_hanging_bricks = ''
}
```

- Include a checkbox option by providing the field (any value). Omit to disable.
- Results go to `static/results/<filename_without_ext>/`.