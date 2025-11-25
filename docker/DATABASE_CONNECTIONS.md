# Database Connection Guide

## 📊 DBeaver Connection Details for PostgreSQL + pgvector

### Connection Parameters

| Parameter | Value |
|-----------|-------|
| **Host** | `localhost` |
| **Port** | `5433` |
| **Database** | `ai_vectors` |
| **Username** | `postgres` |
| **Password** | `minirag` |
| **Driver** | PostgreSQL |

---

## 🔧 Step-by-Step DBeaver Setup

### 1. Create New Connection
1. Open DBeaver
2. Click **Database** → **New Database Connection**
3. Select **PostgreSQL**
4. Click **Next**

### 2. Configure Connection Settings

**Main Tab:**
```
Host:     localhost
Port:     5433
Database: ai_vectors
Username: postgres
Password: minirag
```

**✅ Check "Show all databases"** (optional, to see other databases)

### 3. Test Connection
1. Click **Test Connection** button
2. If first time, DBeaver will download PostgreSQL drivers
3. You should see: **"Connected"** ✅

### 4. Advanced Settings (Optional)

**Driver Properties Tab:**
- You can add custom properties if needed
- For pgvector, no special configuration required

**SSH Tab:**
- Not needed for local connections

---

## 🚀 Quick Start Commands

### Start Database
```bash
cd C:\Users\saeid\Rag-System-Project
docker compose -f docker\docker-compose.yml up -d
```

### Stop Database
```bash
docker compose -f docker\docker-compose.yml down
```

### Reset Database (⚠️ Deletes all data!)
```bash
docker compose -f docker\docker-compose.yml down -v
docker compose -f docker\docker-compose.yml up -d
```

### Check Database Status
```bash
docker compose -f docker\docker-compose.yml ps
```

### View Logs
```bash
docker compose -f docker\docker-compose.yml logs vector_db
```

---

## 🔍 Verify pgvector Extension

After connecting in DBeaver, run this query to verify pgvector is installed:

```sql
-- Check if pgvector extension is available
SELECT * FROM pg_available_extensions WHERE name = 'vector';

-- Enable pgvector extension (run once)
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify extension is enabled
SELECT * FROM pg_extension WHERE extname = 'vector';
```

---

## 🗄️ MongoDB Connection (if needed)

| Parameter | Value |
|-----------|-------|
| **Host** | `localhost` |
| **Port** | `27017` |
| **Username** | `root` |
| **Password** | `example` |
| **Auth Database** | `admin` |

**Connection String:**
```
mongodb://root:example@localhost:27017/?authSource=admin
```

---

## 📝 Common Connection Issues

### Issue 1: "Connection refused"
**Solution:** Make sure Docker containers are running
```bash
docker compose -f docker\docker-compose.yml ps
```

### Issue 2: "Password authentication failed"
**Solution:** Reset the database
```bash
docker compose -f docker\docker-compose.yml down -v
docker compose -f docker\docker-compose.yml up -d
```
Wait 10 seconds for initialization, then try again.

### Issue 3: "Port already in use"
**Solution:** Change port in `docker-compose.yml`
```yaml
ports:
  - "5434:5432"  # Use 5434 instead of 5433
```

---

## 🧪 Test Connection from Command Line

### Using psql (from container)
```bash
docker compose -f docker\docker-compose.yml exec vector_db psql -U postgres -d ai_vectors
```

### Using psql (from Windows with PostgreSQL installed)
```bash
psql -h localhost -p 5433 -U postgres -d ai_vectors
```

### Using Python
```python
import psycopg2

conn = psycopg2.connect(
    host='localhost',
    port=5433,
    user='postgres',
    password='minirag',
    database='ai_vectors'
)
print("✅ Connected successfully!")
conn.close()
```

---

## 📋 Environment Variables Summary

All credentials are stored in: `docker/.env`

```env
# MongoDB
MONGO_INITDB_ROOT_USERNAME=root
MONGO_INITDB_ROOT_PASSWORD=example

# PostgreSQL + pgvector
VECTOR_PGUSER=postgres
VECTOR_PGPASSWORD=minirag
VECTOR_PGDB=ai_vectors
```

**⚠️ Security Note:** These are development credentials. Use strong passwords in production!

---

## 🔐 Change Passwords

To change passwords:

1. Edit `docker/.env` file
2. Recreate containers:
```bash
docker compose -f docker\docker-compose.yml down -v
docker compose -f docker\docker-compose.yml up -d
```

---

## ✅ Checklist

- [ ] Docker Desktop is running
- [ ] Containers are started (`docker compose up -d`)
- [ ] DBeaver is installed
- [ ] PostgreSQL driver is downloaded in DBeaver
- [ ] Connection tested successfully
- [ ] pgvector extension is enabled

---

**Last Updated:** October 28, 2025
