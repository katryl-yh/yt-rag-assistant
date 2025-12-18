from backend.constants import VECTOR_DATABASE_PATH, DATA_PATH
import lancedb
from backend.data_models import Transcript
import time
from pathlib import Path


def setup_vector_db(path):
    Path(path).mkdir(exist_ok=True)
    vector_db = lancedb.connect(uri=path)
    vector_db.create_table("transcripts", schema=Transcript, exist_ok=True)

    return vector_db


def ingest_mds_to_vector_db(table):
    for file in DATA_PATH.glob("*.md"):
        with open(file, "r") as f:
            content = f.read()

        md_id = file.stem
        table.delete(f"md_id = '{md_id}'")

        table.add(
            [
                {
                    "md_id": md_id,
                    "filepath": str(file),
                    "filename": file.stem,
                    "content": content,
                }
            ]
        )

        print(table.to_pandas().shape)
        print(table.to_pandas()["filename"])
        time.sleep(30)


if __name__ == "__main__":
    vector_db = setup_vector_db(VECTOR_DATABASE_PATH)

    ingest_mds_to_vector_db(vector_db["transcripts"])