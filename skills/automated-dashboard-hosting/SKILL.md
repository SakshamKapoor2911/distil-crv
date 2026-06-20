---
name: automated-dashboard-hosting
description: Pipeline for tracking ML research metrics by parsing JSON logs into an interactive, Mermaid-enabled static dashboard automatically hosted on GitHub Pages.
---

# Automated Dashboard Hosting

This skill documents how to create transparent, zero-maintenance research dashboards (similar to those used in Tim-MGDT, Feedback Distillation, and Distil-CRV). The dashboard automatically parses logged experiment data and deploys it dynamically for external viewers.

## 1. The Log-Driven Architecture
Do not hardcode metrics into HTML. Instead, structure your Python training scripts to output JSON payloads to a dedicated directory (e.g., `experiments/results/phase1/baseline.json`).

## 2. The Python Generator (`build_dashboard.py`)
Write a lightweight Python script that executes locally or in CI to compile the static site:
1. **Parse JSONs:** Use `glob` and `json` to load all metric files.
2. **Build HTML Tables:** Format the loaded data into HTML tables.
3. **Embed Raw Data Details:** For maximum transparency, embed the raw JSON payload natively into the HTML under `<details>` tags:
   ```html
   <details style="margin-top: 15px;">
       <summary style="cursor: pointer; color: #58a6ff;">View Raw JSON Data</summary>
       <pre style="background: #0d1117; padding: 10px; border-radius: 5px; overflow-x: auto;">{json.dumps(data, indent=2)}</pre>
   </details>
   ```
4. **Mermaid Diagrams:** Embed architecture flows by including the Mermaid JS CDN in the header and placing syntax inside `<div class="mermaid">`.

## 3. GitHub Actions CI/CD (Bypassing 403 Errors)
A common pitfall when automating dashboards is attempting to make the GitHub Actions Bot run `git push` to save the `index.html` back to the `master` branch. This often fails with a `403 Forbidden` error due to restricted repository token permissions.

*The Solution:* **Do not commit generated UI assets to the master branch.** 
Instead, use the native `actions/upload-pages-artifact` workflow to securely push the `dashboard/` directory directly to the GitHub Pages environment.

**Working `dashboard.yml` configuration:**
```yaml
name: Deploy Dashboard to GitHub Pages

on:
  push:
    branches: ["master"]
  workflow_dispatch:

permissions:
  contents: write
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          
      - name: Build Dashboard
        run: python scripts/build_dashboard.py
        
      # Notice there is NO `git push` step here!
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: './dashboard'

  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

By decoupling the build generation from the repository commit history, the dashboard stays continually updated without causing recursive git loops or permission failures.
