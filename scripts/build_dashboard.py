import os
import json
import glob
import datetime

DASHBOARD_DIR = "dashboard"
RESULTS_DIR = "experiments/results"

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Distil-CRV Dashboard</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; }
  .container { max-width: 900px; margin: 0 auto; }
  h1 { color: #58a6ff; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 20px; margin-bottom: 20px; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 8px; border-bottom: 1px solid #30363d; }
  th { color: #8b949e; }
  .metric { font-size: 24px; font-weight: bold; color: #3fb950; }
</style>
</head>
<body>
<div class="container">
  <h1>Distil-CRV: Dashboard</h1>
  <p>Real-time profiling and accuracy metrics for the Distil-CRV framework.</p>
  
  <div class="card">
    <h2>Phase 1: Baseline VRAM Profiling</h2>
    {phase1_html}
  </div>
</div>
</body>
</html>
"""

def build_dashboard():
    os.makedirs(DASHBOARD_DIR, exist_ok=True)
    
    # Phase 1
    phase1_html = "<p>No runs completed yet.</p>"
    phase1_files = glob.glob(os.path.join(RESULTS_DIR, "phase1", "*.json"))
    
    if phase1_files:
        latest = max(phase1_files, key=os.path.getmtime)
        with open(latest, 'r') as f:
            data = json.load(f)
            
        phase1_html = f"""
        <table>
            <tr><th>Model</th><td>{data.get('model', 'N/A')}</td></tr>
            <tr><th>VRAM Usage (GB)</th><td class="metric">{data.get('baseline_vram_gb', 'N/A')}</td></tr>
            <tr><th>Latency (ms/step)</th><td>{data.get('baseline_latency_ms', 'N/A')}</td></tr>
            <tr><th>Accuracy</th><td>{data.get('accuracy', 0)*100:.1f}%</td></tr>
            <tr><th>Subset Size</th><td>{data.get('subset_size', 'N/A')}</td></tr>
        </table>
        """
        
    html = HTML_TEMPLATE.format(phase1_html=phase1_html)
    
    with open(os.path.join(DASHBOARD_DIR, "index.html"), "w") as f:
        f.write(html)
        
    print("Dashboard built successfully in dashboard/index.html")

if __name__ == "__main__":
    build_dashboard()
