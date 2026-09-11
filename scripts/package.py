"""Build the light client wheel and a source bundle for remote GPU deployment."""
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
subprocess.run(["uv", "build", "--wheel"], cwd=ROOT, check=True)
archive = ROOT / "dist/indextts-agent-0.1.0-source.zip"
files = [ROOT / name for name in ("pyproject.toml", "README.md", ".gitignore", ".dockerignore", "Dockerfile",
                                  "compose.yaml", "upstream.lock.json", "models.lock.json")]
for directory in ("src", "scripts", "tests", "docs"):
    files.extend(p for p in (ROOT / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
    for file in sorted(files):
        bundle.write(file, file.relative_to(ROOT))
print(archive)
