from database import engine, Base
import models  # noqa: F401 — registers all models with Base

Base.metadata.create_all(bind=engine)
print("Database initialized.")
