import duckdb

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("INSTALL spatial; LOAD spatial;")

# Disable SSL verification
con.execute("SET s3_region='us-west-2';")
con.execute("SET s3_url_style='path';")

result = con.execute("""
    SELECT COUNT(*) 
    FROM read_parquet(
        'https://us-west-2.opendata.source.coop/google-research-open-buildings/geoparquet-by-country/country_iso=IDN/*.parquet'
    )
""").fetchone()

print(f"Total bangunan Indonesia: {result[0]:,}")