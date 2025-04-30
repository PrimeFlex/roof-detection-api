# roof_api.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from shapely.geometry import Polygon
import io
import base64
import json

app = FastAPI()
API_KEY = "your-secret-api-key"

# Constants
PIXELS_PER_FOOT = 0.4
FEET_PER_PIXEL = 1 / PIXELS_PER_FOOT


def process_roof_image(image_bytes: bytes):
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Resize for consistency
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 200]

    overlay = image.copy()
    roof_data = []

    for idx, cnt in enumerate(contours):
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        cv2.drawContours(overlay, [approx], -1, (0, 255, 0), 2)

        edges_ft = []
        for i in range(len(approx)):
            pt1 = approx[i][0]
            pt2 = approx[(i + 1) % len(approx)][0]
            dist_px = np.linalg.norm(pt1 - pt2)
            dist_ft = dist_px * FEET_PER_PIXEL
            edges_ft.append(round(dist_ft, 2))
            mid = tuple(((pt1 + pt2) // 2).astype(int))
            cv2.putText(overlay, f"{dist_ft:.1f} ft", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        poly = Polygon([tuple(p[0]) for p in approx])
        area_ft2 = poly.area * (FEET_PER_PIXEL ** 2)

        roof_data.append({
            "id": idx,
            "vertices": [tuple(map(int, p[0])) for p in approx],
            "area_sqft": round(area_ft2, 2),
            "edges_ft": edges_ft
        })

    # Encode overlay image
    _, buffer = cv2.imencode(".png", overlay)
    encoded_img = base64.b64encode(buffer).decode("utf-8")

    return {
        "image_base64": encoded_img,
        "scale": PIXELS_PER_FOOT,
        "roofs": roof_data
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...), x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

    contents = await file.read()
    try:
        result = process_roof_image(contents)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
