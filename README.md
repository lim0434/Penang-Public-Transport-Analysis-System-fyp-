# Penang-Public-Transport-Analysis-System-fyp-

## PostgreSQL setup

1. Copy `.env.example` to `.env` and enter the PostgreSQL password from pgAdmin.
2. Install the database connector with `pip install -r requirements-db.txt`.
3. Start PostgreSQL, then run `python src/migrate_to_postgres.py` once.
4. Start the dashboard as usual. The original data files remain as backup copies.
