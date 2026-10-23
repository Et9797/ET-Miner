#!/usr/bin/env python3
"""Export uniprotAccession,globalMetricValue from the AlphaFold BigQuery
metadata table to CSV through the BigQuery Storage Read API.

Reads the public table `bigquery-public-data.deepmind_alphafold.metadata`
with the two selected columns over parallel Arrow streams, using the gcloud
user access token passed on the command line, and writes one CSV with the
header `uniprotAccession,globalMetricValue` (the layout af-extract expects).

Usage:
    export_plddt_storage_api.py OUTPUT_CSV --token TOKEN [--project ID]
                                [--streams N]

Options:
    --token     OAuth2 access token (e.g. from `gcloud auth print-access-token`)
    --project   billing project for the read session (default et-research-491307)
    --streams   requested parallel read streams (default 16)
"""

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.csv as pacsv
from google.cloud import bigquery_storage
from google.cloud.bigquery_storage import types
from google.oauth2.credentials import Credentials

TABLE = "projects/bigquery-public-data/datasets/deepmind_alphafold/tables/metadata"
COLUMNS = ["uniprotAccession", "globalMetricValue"]


def read_stream(client, session, stream, out, lock, counter):
    """Read one stream and append its batches to ``out`` as CSV rows."""
    reader = client.read_rows(stream.name)
    for batch in _batches(reader, session):
        batch = batch.select(COLUMNS)
        buf = pa.BufferOutputStream()
        pacsv.write_csv(batch, buf, write_options=pacsv.WriteOptions(
            include_header=False, quoting_style="none"))
        data = buf.getvalue().to_pybytes()
        with lock:
            out.write(data)
            counter[0] += batch.num_rows


def _batches(reader, session):
    """Yield the Arrow record batches of one read stream."""
    for page in reader.rows(session).pages:
        yield page.to_arrow()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("output")
    ap.add_argument("--token", required=True)
    ap.add_argument("--project", default="et-research-491307")
    ap.add_argument("--streams", type=int, default=16)
    a = ap.parse_args()

    client = bigquery_storage.BigQueryReadClient(credentials=Credentials(token=a.token))
    opts = types.ReadSession.TableReadOptions(selected_fields=COLUMNS)
    req = types.ReadSession(table=TABLE, data_format=types.DataFormat.ARROW, read_options=opts)
    session = client.create_read_session(parent=f"projects/{a.project}", read_session=req,
                                         max_stream_count=a.streams)
    print(f"streams={len(session.streams)} est_bytes={session.estimated_total_bytes_scanned}", flush=True)
    lock, counter, t0 = threading.Lock(), [0], time.time()
    with open(a.output, "wb") as out:
        out.write((",".join(COLUMNS) + "\n").encode())
        with ThreadPoolExecutor(max_workers=len(session.streams)) as pool:
            futs = [pool.submit(read_stream, client, session, s, out, lock, counter)
                    for s in session.streams]
            while any(not f.done() for f in futs):
                time.sleep(15)
                print(f"rows={counter[0]:,} elapsed={time.time()-t0:.0f}s", flush=True)
            for f in futs:
                f.result()
    print(f"done rows={counter[0]:,} elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
