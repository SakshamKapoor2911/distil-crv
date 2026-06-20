import re
import os

files = [
    'dashboard/highlights.html',
    'docs/phase3_highlights.html'
]

for filepath in files:
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            html = f.read()
        
        # Replace the literal `\` token and `n` token with an HTML line break
        # The tokens look like: <span class="token" style="...">\</span><span class="token" style="...">n</span>
        # But sometimes the `\` is attached to the previous token, e.g. `?\</span>` or `.\\\</span>`
        
        # 1. First, replace the explicit `\</span><span class="token" style="[^"]*">n</span>` with `</span><br>`
        html = re.sub(r'\\</span><span class="token" style="[^"]*">n</span>', r'</span><br>', html)
        
        # 2. If there are any stray `\n` literal strings inside a token, replace them too
        html = html.replace(r'\n', '<br>')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"Fixed {filepath}")
