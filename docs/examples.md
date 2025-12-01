# Examples & Recipes — focused

This page collects short, copy-pasteable examples sorted by audience and use-case.

For full examples and longer recipes, see the `examples/` folder or run the CLI help.

## User quickstarts

1. Fast general conversion (good default)

```bash
vectalab convert input.png
```

1. Production-quality (best results, uses SVGO if available)

```bash
vectalab premium input.png
```

1. Optimize an existing SVG

```bash
vectalab optimize icon.svg -p 1
```

## Developer / Integrator recipes

1. Batch processing (parallel)

```python
from vectalab import vectorize_premium
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def process(path):
    out = Path(path).with_suffix('.svg')
    vectorize_premium(str(path), str(out), verbose=False)

with ThreadPoolExecutor(max_workers=4) as ex:
    list(ex.map(process, Path('images').glob('*.png')))
```

1. Web integration (FastAPI minimal)

```python
from fastapi import FastAPI, UploadFile
from vectalab import vectorize_premium
import tempfile

app = FastAPI()

@app.post('/vectorize')
async def vectorize(file: UploadFile):
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        out = tmp.name.replace('.png', '.svg')
        svg, m = vectorize_premium(tmp.name, out)
        return {'svg': open(svg).read(), 'metrics': m}
```

## Quality & tuning tips (short)

- Precision (-p): 1 -> agressive file size reduction; 2 -> balanced; 3+ -> higher fidelity
- Colors (-c): small palettes (8–16) for logos, larger for photos
- SVGO: big win on file size; run `vectalab svgo-info` to check availability

If you want longer, focused recipes, say which audience you care about (end-users, API integrators, researchers) and I will expand with a step-by-step example tuned to your needs.
