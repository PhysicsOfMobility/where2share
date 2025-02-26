# Where2Share

## Setup

- Recommended python version: 3.12
- Install graph-tool
- Clone the repository:
```bash
git clone git@github.com:PhysicsOfMobility/where2share.git
cd where2share
pip install -e .[dev,doc]
sphinx-build -W -j7 -b html doc/source doc/_build/html
pre-commit install
```
- Alternatively, `pip`-install directly: 
```bash
pip install "where2share@git+https://github.com/PhysicsOfMobility/where2share"
```
- Create a file called `.env` in the repository root to configure the DB connection:
```ini
DB_HOST=...
DB_PORT=...
DB_USER=...
DB_PASS=...
DB_NAME=...
```
- Test the installation:
```bash
pytest
```

To test only "fast" tests, use:
```bash
pytest -m "not slow"
```

## Documentation

After building it (during setup), just open `doc/_build/html/index.html` in your browser
of choice.
