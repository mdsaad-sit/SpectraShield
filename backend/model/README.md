# Trained Model Files

Place the trained model files here:

- `best_model.pth`
- `model_config.json`

These files are intentionally excluded from Git.

After placing them:

```bash
restart the backend.
```

Then:

```bash
GET /api/health
```

should show:

```json
{
    "model_loaded": true
}
```