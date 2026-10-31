WildWatch dashboard

Run it with: powershell -ExecutionPolicy Bypass -File src/scripts/run.ps1 dashboard
src/server.py serves these pages at http://localhost:8000 with real data from the database, docs/reports/ and src/config.yaml.
Pages: Main (home), Upload, Alerts, Analytics, Accuracy.
wildwatch-runtime.js renders the page templates. wildwatch-api.js calls the API.
Opening the files directly will not work: every number comes from the API, and missing values show "Not measured".
