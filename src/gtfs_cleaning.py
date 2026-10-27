from pathlib import Path
import pandas as pd

# GTFS folder
GTFS_DIR = Path(r"D:\fyp\data\raw\gtfs")

files = [
    "agency.txt",
    "calendar.txt",
    "routes.txt",
    "trips.txt",
    "stop_times.txt",
    "stops.txt",
    "shapes.txt"
]

print("=" * 60)
print("GTFS DATA INSPECTION")
print("=" * 60)

for file in files:
    file_path = GTFS_DIR / file

    print(f"\n--- {file} ---")

    if not file_path.exists():
        print("FILE NOT FOUND")
        continue

    df = pd.read_csv(file_path)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    print(list(df.columns))

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

print("\n" + "=" * 60)
print("INSPECTION COMPLETED")
print("=" * 60)


# ---------------------------------------------------------
# GTFS RELATIONSHIP VALIDATION
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("GTFS RELATIONSHIP VALIDATION")
print("=" * 60)

# Load data
agency = pd.read_csv(GTFS_DIR / "agency.txt")
calendar = pd.read_csv(GTFS_DIR / "calendar.txt")
routes = pd.read_csv(GTFS_DIR / "routes.txt")
trips = pd.read_csv(GTFS_DIR / "trips.txt")
stop_times = pd.read_csv(GTFS_DIR / "stop_times.txt")
stops = pd.read_csv(GTFS_DIR / "stops.txt")
shapes = pd.read_csv(GTFS_DIR / "shapes.txt")


# 1. route_id
invalid_routes = trips[
    ~trips["route_id"].isin(routes["route_id"])
]

print(f"\nInvalid route_id in trips: {len(invalid_routes):,}")


# 2. service_id
invalid_services = trips[
    ~trips["service_id"].isin(calendar["service_id"])
]

print(f"Invalid service_id in trips: {len(invalid_services):,}")


# 3. shape_id
invalid_shapes = trips[
    ~trips["shape_id"].isin(shapes["shape_id"])
]

print(f"Invalid shape_id in trips: {len(invalid_shapes):,}")


# 4. trip_id
invalid_trips = stop_times[
    ~stop_times["trip_id"].isin(trips["trip_id"])
]

print(f"Invalid trip_id in stop_times: {len(invalid_trips):,}")


# 5. stop_id
invalid_stops = stop_times[
    ~stop_times["stop_id"].isin(stops["stop_id"])
]

print(f"Invalid stop_id in stop_times: {len(invalid_stops):,}")


# 6. Duplicate IDs
print("\nDuplicate ID checks:")

print(
    "Duplicate route_id:",
    routes["route_id"].duplicated().sum()
)

print(
    "Duplicate trip_id:",
    trips["trip_id"].duplicated().sum()
)

print(
    "Duplicate stop_id:",
    stops["stop_id"].duplicated().sum()
)

print(
    "Duplicate shape_id + sequence:",
    shapes.duplicated(
        subset=["shape_id", "shape_pt_sequence"]
    ).sum()
)

print("\n" + "=" * 60)
print("VALIDATION COMPLETED")
print("=" * 60)


# ---------------------------------------------------------
# GTFS DATA QUALITY CHECKS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("GTFS DATA QUALITY CHECKS")
print("=" * 60)


# ---------------------------------------------------------
# 1. STOP COORDINATE CHECK
# ---------------------------------------------------------

print("\n--- Stop Coordinate Check ---")

invalid_lat = stops[
    (stops["stop_lat"] < -90) |
    (stops["stop_lat"] > 90)
]

invalid_lon = stops[
    (stops["stop_lon"] < -180) |
    (stops["stop_lon"] > 180)
]

print(f"Invalid latitude: {len(invalid_lat):,}")
print(f"Invalid longitude: {len(invalid_lon):,}")


# Check for missing coordinates
missing_coordinates = stops[
    stops["stop_lat"].isna() |
    stops["stop_lon"].isna()
]

print(f"Missing coordinates: {len(missing_coordinates):,}")


# ---------------------------------------------------------
# 2. STOP SEQUENCE CHECK
# ---------------------------------------------------------

print("\n--- Stop Sequence Check ---")

# Duplicate stop sequence within the same trip
duplicate_sequences = stop_times[
    stop_times.duplicated(
        subset=["trip_id", "stop_sequence"],
        keep=False
    )
]

print(
    "Duplicate stop sequences within trips:",
    len(duplicate_sequences)
)


# Check for missing stop sequence
missing_sequence = stop_times[
    stop_times["stop_sequence"].isna()
]

print(
    "Missing stop sequence:",
    len(missing_sequence)
)


# ---------------------------------------------------------
# 3. TIME FORMAT CHECK
# ---------------------------------------------------------

print("\n--- Time Check ---")

def check_gtfs_time(time_value):
    try:
        parts = str(time_value).split(":")

        if len(parts) != 3:
            return False

        hour = int(parts[0])
        minute = int(parts[1])
        second = int(parts[2])

        # GTFS allows hours greater than 23
        if hour < 0:
            return False

        if minute < 0 or minute > 59:
            return False

        if second < 0 or second > 59:
            return False

        return True

    except:
        return False


invalid_arrival = stop_times[
    ~stop_times["arrival_time"].apply(check_gtfs_time)
]

invalid_departure = stop_times[
    ~stop_times["departure_time"].apply(check_gtfs_time)
]

print(
    "Invalid arrival_time:",
    len(invalid_arrival)
)

print(
    "Invalid departure_time:",
    len(invalid_departure)
)


# ---------------------------------------------------------
# 4. ARRIVAL / DEPARTURE CONSISTENCY
# ---------------------------------------------------------

print("\n--- Arrival and Departure Check ---")

def time_to_seconds(time_value):
    parts = str(time_value).split(":")

    hour = int(parts[0])
    minute = int(parts[1])
    second = int(parts[2])

    return hour * 3600 + minute * 60 + second


stop_times["arrival_seconds"] = (
    stop_times["arrival_time"].apply(time_to_seconds)
)

stop_times["departure_seconds"] = (
    stop_times["departure_time"].apply(time_to_seconds)
)

invalid_time_order = stop_times[
    stop_times["departure_seconds"] <
    stop_times["arrival_seconds"]
]

print(
    "Departure earlier than arrival:",
    len(invalid_time_order)
)


# ---------------------------------------------------------
# 5. TRIP STOP SEQUENCE ORDER
# ---------------------------------------------------------

print("\n--- Trip Sequence Order Check ---")

sequence_difference = (
    stop_times
    .sort_values(["trip_id", "stop_sequence"])
    .groupby("trip_id")["stop_sequence"]
    .diff()
)

invalid_sequence_order = sequence_difference[
    sequence_difference <= 0
]

print(
    "Non-increasing stop sequences:",
    len(invalid_sequence_order)
)


# ---------------------------------------------------------
# 6. CALENDAR DATE CHECK
# ---------------------------------------------------------

print("\n--- Calendar Date Check ---")

calendar["start_date"] = pd.to_datetime(
    calendar["start_date"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

calendar["end_date"] = pd.to_datetime(
    calendar["end_date"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

invalid_dates = calendar[
    calendar["start_date"].isna() |
    calendar["end_date"].isna()
]

print(
    "Invalid calendar dates:",
    len(invalid_dates)
)


# Check start date after end date
invalid_date_range = calendar[
    calendar["start_date"] > calendar["end_date"]
]

print(
    "Start date after end date:",
    len(invalid_date_range)
)


# ---------------------------------------------------------
# 7. SERVICE DAY CHECK
# ---------------------------------------------------------

print("\n--- Service Day Check ---")

day_columns = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday"
]

no_service_day = calendar[
    calendar[day_columns].sum(axis=1) == 0
]

print(
    "Service records with no active day:",
    len(no_service_day)
)


# ---------------------------------------------------------
# 8. SHAPE SEQUENCE CHECK
# ---------------------------------------------------------

print("\n--- Shape Sequence Check ---")

duplicate_shape_sequence = shapes[
    shapes.duplicated(
        subset=["shape_id", "shape_pt_sequence"]
    )
]

print(
    "Duplicate shape sequences:",
    len(duplicate_shape_sequence)
)

missing_shape_sequence = shapes[
    shapes["shape_pt_sequence"].isna()
]

print(
    "Missing shape sequences:",
    len(missing_shape_sequence)
)


# ---------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("GTFS DATA QUALITY CHECK COMPLETED")
print("=" * 60)