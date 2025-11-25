
# Alembic Migrations Guide

## Working Directory

Execute all commands from:
```
/src/models/db_schemas/rag
```

## Configuration

1. **Copy the example configuration file:**
    ```bash
    cp alembic.ini.example alembic.ini
    ```

2. **Update database credentials:**
    Edit `alembic.ini` and update the `sqlalchemy.url` with your database connection string.

## Creating Migrations (Optional)

To create a new migration with auto-generated changes:

```bash
alembic revision --autogenerate -m "Add [description]"
```

## Running Migrations

Apply all pending migrations to upgrade your database:

```bash
alembic upgrade head
```

