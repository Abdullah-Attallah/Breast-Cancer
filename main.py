from fastapi import FastAPI,Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = joblib.load("SVCModel.pkl")

@app.get("/", response_class=HTMLResponse)
async def serve_html():
    with open("index.html", "r") as file:
        return file.read()
@app.get("/predict")
async def predict(
        radius_mean: float = Query(..., title="22 mm"),
        texture_mean: float = Query(..., title="Texture Mean"),
        perimeter_mean: float = Query(..., title="Perimeter Mean"),
        area_mean: float = Query(..., title="Area Mean"),
        smoothness_mean: float = Query(..., title="Smoothness Mean"),
        compactness_mean: float = Query(..., title="Compactness Mean"),
        concavity_mean: float = Query(..., title="Concavity Mean"),
        concave_points_mean: float = Query(..., title="Concave Points Mean"),
        symmetry_mean: float = Query(..., title="Symmetry Mean"),
        fractal_dimension_mean: float = Query(..., title="Fractal Dimension Mean"),
        radius_se: float = Query(..., title="Radius SE"),
        texture_se: float = Query(..., title="Texture SE"),
        perimeter_se: float = Query(..., title="Perimeter SE"),
        area_se: float = Query(..., title="Area SE"),
        smoothness_se: float = Query(..., title="Smoothness SE"),
        compactness_se: float = Query(..., title="Compactness SE"),
        concavity_se: float = Query(..., title="Concavity SE"),
        concave_points_se: float = Query(..., title="Concave Points SE"),
        symmetry_se: float = Query(..., title="Symmetry SE"),
        fractal_dimension_se: float = Query(..., title="Fractal Dimension SE"),
        radius_worst: float = Query(..., title="Radius Worst"),
        texture_worst: float = Query(..., title="Texture Worst"),
        perimeter_worst: float = Query(..., title="Perimeter Worst"),
        area_worst: float = Query(..., title="Area Worst"),
        smoothness_worst: float = Query(..., title="Smoothness Worst"),
        compactness_worst: float = Query(..., title="Compactness Worst"),
        concavity_worst: float = Query(..., title="Concavity Worst"),
        concave_points_worst: float = Query(..., title="Concave Points Worst"),
        symmetry_worst: float = Query(..., title="Symmetry Worst"),
        fractal_dimension_worst: float = Query(..., title="Fractal Dimension Worst"),

):
    data = {
        "radius_mean": radius_mean,
        "texture_mean": texture_mean,
        "perimeter_mean": perimeter_mean,
        "area_mean": area_mean ,
        "smoothness_mean": smoothness_mean,
        "compactness_mean": compactness_mean,
        "concavity_mean": concavity_mean,
        "concave_points_mean": concave_points_mean,
        "symmetry_mean": symmetry_mean,
        "fractal_dimension_mean": fractal_dimension_mean,
        "radius_se": radius_se,
        "texture_se": texture_se,
        "perimeter_se": perimeter_se,
        "area_se": area_se,
        "smoothness_se": smoothness_se,
        "compactness_se": compactness_se,
        "concavity_se": concavity_se,
        "concave_points_se": concave_points_se,
        "symmetry_se": symmetry_se,
        "fractal_dimension_se": fractal_dimension_se,
        "radius_worst": radius_worst,
        "texture_worst": texture_worst,
        "perimeter_worst": perimeter_worst,
        "area_worst": area_worst,
        "smoothness_worst": smoothness_worst,
        "compactness_worst": compactness_worst,
        "concavity_worst": concavity_worst,
        "concave_points_worst": concave_points_worst,
        "symmetry_worst": symmetry_worst,
        "fractal_dimension_worst": fractal_dimension_worst,
    }

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(df)

    # Get classification result
    classification_map = {0: "Malignant", 1: "Benign"}
    classification = classification_map.get(prediction[0], "Unknown")

    return {"prediction": int(prediction[0]), "classification": classification}


@app.post("/")
async def post():
    return {"message": "Hello"}

@app.put("/")
async def put():
    return {"message": "Hello"}
