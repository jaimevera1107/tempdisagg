# Example data

`legacy_long_format.csv` is the example file shipped with tempdisagg 0.2.x, in the
long format (`Index`, `Grain`, `y`, `X`). It is kept because it is a realistic
messy input: the first and last years are incomplete and the target changes
within some years. `examples/quickstart.py` shows how the library rejects it
and how to clean it explicitly.

The real datasets used in the documentation ship with the package:

```python
from tempdisagg.datasets import load_swisspharma, load_denton1971
sales, trade = load_swisspharma()     # annual sales, quarterly exports and imports
totals, indicator = load_denton1971() # textbook example of Denton (1971)
```

The CSV files are in `tempdisagg/datasets/data/`.
