import ast
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import networkx as nx
import math

p = Path(r"D:\fyp\src\app.py")
f = next(n for n in ast.parse(p.read_text(encoding="utf-8")).body
         if isinstance(n, ast.FunctionDef) and n.name == "_spearman_rho")
ns = {"pd": pd}
exec(compile(ast.Module(body=[f], type_ignores=[]), str(p), "exec"), ns)
# No.1 Spearman correlation
result = ns["_spearman_rho"](pd.Series([1, 2, 3]), pd.Series([2, 4, 6]))
assert result == 1.0
print(f"PASS: {result:.3f}")
# No.2 Spearman correlation
result = ns["_spearman_rho"](pd.Series([1, 2, 3]), pd.Series([6, 4, 2]))
assert result == -1.0
print(f"PASS: {result:.3f}")
# No.3 Spearman correlation 
result = ns["_spearman_rho"](pd.Series([1, 2, None, 4]), pd.Series([2, 4, 6, 8]))
assert result == 1.0
print(f"PASS: {result:.3f}")
# No.4 Spearman correlation
result = ns["_spearman_rho"](pd.Series([1, 1, 1]), pd.Series([2, 3, 4]))
assert pd.isna(result)
print("PASS: no valid correlation")

# No.5 Min-Max scaling
p = Path(r"D:\fyp\src\tai_by_mukim.py")
f = next(n for n in ast.parse(p.read_text(encoding="utf-8")).body
         if isinstance(n, ast.FunctionDef) and n.name == "minmax_score")
ns = {"pd": pd, "np": np}
exec(compile(ast.Module(body=[f], type_ignores=[]), str(p), "exec"), ns)

result = ns["minmax_score"](pd.Series([10, 20, 30]))
assert result.tolist() == [0.0, 0.5, 1.0]
print(f"PASS: {result.tolist()}")
# No.6 Min-Max scaling with reverse
result = ns["minmax_score"](pd.Series([10, 20, 30]), reverse=True)
assert result.tolist() == [1.0, 0.5, 0.0]
print(f"PASS: {result.tolist()}")
# No.7 Min-Max scaling with equal values
result = ns["minmax_score"](pd.Series([5, 5]))
assert result.tolist() == [0.5, 0.5]
print(f"PASS: {result.tolist()}")
# No.8 Min-Max scaling with missing values
result = ns["minmax_score"](pd.Series([10, np.nan, 30]))
assert result.iloc[0] == 0.0 and pd.isna(result.iloc[1]) and result.iloc[2] == 1.0
print("PASS: valid scores scaled, missing value preserved")

# No.9 TAI calculation
p = Path(r"D:\fyp\src\tai_by_mukim.py")
tree = ast.parse(p.read_text(encoding="utf-8"))
stmt = next(
    n for n in ast.walk(tree)
    if isinstance(n, ast.Assign)
    and isinstance(n.targets[0], ast.Subscript)
    and isinstance(n.targets[0].slice, ast.Constant)
    and n.targets[0].slice.value == "TAI"
)
cols = ["stop_density_score", "population_coverage_score",
        "walking_distance_score", "destination_access_score"]
tai = pd.DataFrame([[0.8, 0.6, 0.4, 1.0]], columns=cols)
result = eval(compile(ast.Expression(stmt.value), str(p), "eval"),
              {"tai": tai, "score_columns": cols})
assert result.iloc[0] == 0.7
print(f"PASS: TAI = {result.iloc[0]:.2f}")
# No.10 TAI calculation with missing values
tai = pd.DataFrame([[0.8, 0.6, float("nan"), 1.0]], columns=cols)
result = eval(compile(ast.Expression(stmt.value), str(p), "eval"),
              {"tai": tai, "score_columns": cols})
assert pd.isna(result.iloc[0])
print("PASS: TAI unavailable when a component is missing")

# No.11 400m coverage calculation
p = Path(r"D:\fyp\src\walking_accessibility_400m.py")
tree = ast.parse(p.read_text(encoding="utf-8"))
stmt = max((n for n in ast.walk(tree) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "covered" for t in n.targets)
            and n.lineno > 400), key=lambda n: n.lineno)
distances = np.array([399.9, 400.0, 400.1])
result = eval(compile(ast.Expression(stmt.value), str(p), "eval"),
              {"route_distances": distances, "CUTOFF_M": 400})
assert result.tolist() == [True, True, False]
print(f"PASS: {result.tolist()}")

# No.12 Population coverage percentage
stmt = max((n for n in ast.walk(tree) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "coverage_percent" for t in n.targets)
            and n.lineno > 400), key=lambda n: n.lineno)
result = eval(compile(ast.Expression(stmt.value), str(p), "eval"),
              {"covered_population": 75, "total_population": 100})
assert result == 75.0
print(f"PASS: {result:.1f}%")

# No.13 Silhouette score for separated groups
p = Path(r"D:\fyp\src\clustering_metrics.py")
tree = ast.parse(p.read_text(encoding="utf-8"))
f = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
         and n.name == "_silhouette_from_labels")
ns = {"np": np}
exec(compile(ast.Module(body=[f], type_ignores=[]), str(p), "exec"), ns)

score = ns["_silhouette_from_labels"](
    np.array([[0.0], [0.1], [5.0], [5.1]]), [0, 0, 1, 1]
)
assert round(score, 3) == 0.980
print(f"PASS: silhouette score = {score:.3f}")
# No.14 Silhouette score is not returned for a single group 
score = ns["_silhouette_from_labels"](np.array([[0.0], [1.0], [2.0]]), [0, 0, 0])
assert np.isnan(score)
print("PASS: one group returns no silhouette score")
# No.15 Silhouette scores for candidate cluster counts
sys.path.insert(0, r"D:\fyp\src")
from clustering_metrics import calculate_silhouette_scores, SCORE_COLUMNS
values = [0, 0.1, 0.2, 5, 5.1, 5.2, 10, 10.1, 10.2]
frame = pd.DataFrame({column: values for column in SCORE_COLUMNS})
scores = calculate_silhouette_scores(frame)
assert set(scores) == {2, 3, 4}
assert all(np.isfinite(score) for score in scores.values())
print(f"PASS: {scores}")
# No.16 Silhouette scores missing indicator columns
scores = calculate_silhouette_scores(pd.DataFrame({"TAI": [0.1, 0.2, 0.3]}))
assert scores == {}
print("PASS: missing indicator columns return no scores")

# No.17 Distance calculation
from linkbike_page import _distance_metres

same = _distance_metres(5.0, 100.0, 5.0, 100.0)
step = _distance_metres(5.0, 100.0, 5.001, 100.0)
assert same == 0 and abs(step - 111.2) < 0.5
print(f"PASS: same point = {same:.1f} m; latitude step = {step:.1f} m")
# No.18 LinkBike station data cleaning
sys.path.insert(0, r"D:\fyp\src")
from linkbike_page import _normalise_stations
data = pd.DataFrame({
    "name": ["Included", "Excluded", "Invalid"],
    "lat": [5.4, 5.5, 91.0],
    "lon": [100.3, 100.4, 100.5],
    "analysis_include": ["yes", "no", "yes"],
})
result = _normalise_stations(data)
assert len(result) == 1 and result.iloc[0]["station_name"] == "Included"
print("PASS: valid included station retained")
# No.19 LinkBike station data cleaning rejects missing latitude
try:
    _normalise_stations(pd.DataFrame({"station_name": ["Station"], "latitude": [5.0]}))
except ValueError:
    print("PASS: missing longitude rejected")
else:
    raise AssertionError("Expected a ValueError")
# No.20 LinkBike distance at the 400 m limit
from scipy.spatial import cKDTree
from unittest.mock import patch
import linkbike_page as lb
graph = nx.Graph()
graph.add_node("n0", x=100.0, y=5.0)
graph.add_node("n1", x=100.004, y=5.0)
graph.add_edge("n0", "n1", length=400.0)
nodes = ["n0", "n1"]
coords = np.array([[100.0, 5.0], [100.004, 5.0]])
tree = cKDTree(np.column_stack((coords[:, 0] * math.cos(math.radians(5)), coords[:, 1])))
stations = pd.DataFrame({"station_name": ["S"], "latitude": [5.0], "longitude": [100.0]})
stops = pd.DataFrame({"stop_id": ["B"], "stop_name": ["Stop"], "stop_lat": [5.0],
                      "stop_lon": [100.004], "nearest_node": ["n1"]})

with patch.object(lb, "_load_walking_network", return_value=(graph, nodes, coords, tree, 5.0)):
    _, _, _, result, _ = lb._analyse(stations, stops, 400)

assert result.iloc[0]["status"] == "Within 400 m"
print("PASS: 400 m stop counted within range")
# No.21 LinkBike distance just outside the 399 m limit
with patch.object(lb, "_load_walking_network", return_value=(graph, nodes, coords, tree, 5.0)):
    _, _, _, result, _ = lb._analyse(stations, stops, 399)

assert result.iloc[0]["status"] == "No LinkBike station within 399 m"
print("PASS: 400 m stop outside the 399 m range")

# No.22 Four-zone classification
p = Path(r"D:\fyp\src\linkbike_page.py")
tree = ast.parse(p.read_text(encoding="utf-8"))
stmt = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Subscript)
            and isinstance(n.targets[0].slice, ast.Constant)
            and n.targets[0].slice.value == "zone")
population = pd.DataFrame({
    "bus_access": [True, True, False, False],
    "bike_access": [True, False, True, False],
})
exec(compile(ast.Module(body=[stmt], type_ignores=[]), str(p), "exec"),
     {"population": population, "np": np})
assert population["zone"].tolist() == ["Both", "Bus only", "LinkBike only", "Neither"]
print(f"PASS: {population['zone'].tolist()}")

# No.23 Candidate destination count
import scenario_simulation as sim
coords = np.array([[100.0, 5.0], [100.004, 5.0], [100.008, 5.0]])
tree = cKDTree(np.column_stack((coords[:, 0] * math.cos(math.radians(5)), coords[:, 1])))
graph = nx.Graph()
for node, (lon, lat) in zip(["A", "B", "C"], coords):
    graph.add_node(node, x=lon, y=lat)
graph.add_edge("A", "B", length=400.0)
graph.add_edge("B", "C", length=1.0)
candidate = pd.DataFrame([{"destination_lon": 100.0, "destination_lat": 5.0}])
places = pd.DataFrame([
    {"destination_lon": 100.004, "destination_lat": 5.0, "destination_type": "School"},
    {"destination_lon": 100.008, "destination_lat": 5.0, "destination_type": "Hospital"},
])
result = sim._candidate_destination_counts(graph, candidate, places,
                                           ["A", "B", "C"], coords, tree, 5.0)
assert result[0]["School"] == 1 and result[0]["Hospital"] == 0
print("PASS: 400 m place counted; 401 m place excluded")

# No.24 Scenario network distance
graph = nx.Graph()
graph.add_edge("A", "B", length=100)
graph.add_edge("B", "C", length=200)
result = sim._distances_from_sources(graph, [("A", 10), ("C", 50)])
assert result == {"A": 10, "B": 110, "C": 50}
print(f"PASS: {result}")

# No.25 Distance to the nearest bus stop
from connectivity_analysis import distances_to_nearest_stop

graph = nx.DiGraph()
graph.add_edge(1, 2, length=100)
graph.add_edge(2, 3, length=250)
graph.add_edge(4, 3, length=75)
distances, stops = distances_to_nearest_stop(graph, [1, 4])
assert distances == {1: 0.0, 4: 0.0, 2: 100.0, 3: 75.0}
assert stops[3] == 4
print(f"PASS: nearest stop for node 3 is {stops[3]} at {distances[3]:.0f} m")

# No.26 Scenario distance calculation
from scenario_simulation import _distance_m
same = _distance_m(100.0, 5.0, 100.0, 5.0)
step = _distance_m(100.0, 5.0, 100.0, 5.001)
assert same == 0 and abs(step - 111.2) < 0.5
print(f"PASS: same point = {same:.1f} m; latitude step = {step:.1f} m")

data = [
    {"match_name":"A","bus_stops":1,"area_km2":1,"bus_stops_per_km2":1,"population_coverage_percent":10,
    "mean_walk_distance_m_routable_population":300,"schools_per_stop":0,"hospitals_per_stop":0,"clinics_per_stop":0,"malls_per_stop":0},
    {"match_name":"B","bus_stops":2,"area_km2":1,"bus_stops_per_km2":2,"population_coverage_percent":100,
    "mean_walk_distance_m_routable_population":100,"schools_per_stop":1,"hospitals_per_stop":1,"clinics_per_stop":1,"malls_per_stop":1},
]
counts = [{"School":2,"Hospital":2,"Clinic":2,"Mall":2}]

with patch.object(sim, "read_dataframe", return_value=pd.DataFrame(data)):
    score = sim._scenario_tai_score("A", 1, 55, 200, counts)

assert round(score, 2) == 0.75
print(f"PASS: simulated TAI = {score:.3f}")

# No.27 Recommendation priority ranking
from recommendations import _priority_table

scored = pd.DataFrame([
    {"match_name":"A","TAI_category":"Low","TAI":0.1,"TAI_rank":2,"csv_population":100,"estimated_csv_population_covered":20},
    {"match_name":"B","TAI_category":"Low","TAI":0.2,"TAI_rank":1,"csv_population":200,"estimated_csv_population_covered":150},
    {"match_name":"C","TAI_category":"High","TAI":0.8,"TAI_rank":3,"csv_population":300,"estimated_csv_population_covered":250},
])
places = pd.DataFrame([
    {"status":"Beyond 400 m","mukim":"A"},
    {"status":"Beyond 400 m","mukim":"A"},
    {"status":"Beyond 400 m","mukim":"B"},
])
result = _priority_table(scored, places, {"A":"Area A","B":"Area B"})

assert result["match_name"].tolist() == ["A", "B"]
assert result["Residents outside a short walk"].tolist() == [80, 50]
assert result["Key places beyond 400 m"].tolist() == [2, 1]
print("PASS: lowest-TAI area ranked first; gap counts correct")

# No.28 Combined Recommendations export
from recommendations import _combined_export

frames = [
    pd.DataFrame([{"Result type":"Priority mukim","Mukim":"Area A"}]),
    pd.DataFrame([{"Result type":"Key place review","Place":"School X"}]),
]
result = _combined_export(frames)

assert result["Result type"].tolist() == ["Priority mukim", "Key place review"]
assert result.iloc[0]["Mukim"] == "Area A"
assert result.iloc[1]["Place"] == "School X"
print("PASS: both result types appear in one export")