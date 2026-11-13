import os
import tempfile
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Import your existing functions from merge.py
from api.merge import MergeError, compose, load_image, save_image

app = FastAPI(title="Coin Merger API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/process")
async def process_images(
    slab_img: UploadFile = File(...),
    upper_img: UploadFile = File(...),
    lower_img: UploadFile = File(...),
):
  try:
    # Vercel's file system is read-only except for /tmp
    with tempfile.TemporaryDirectory() as temp_dir:
      slab_path = os.path.join(temp_dir, slab_img.filename)
      upper_path = os.path.join(temp_dir, upper_img.filename)
      lower_path = os.path.join(temp_dir, lower_img.filename)

      with open(slab_path, "wb") as f:
        f.write(await slab_img.read())
      with open(upper_path, "wb") as f:
        f.write(await upper_img.read())
      with open(lower_path, "wb") as f:
        f.write(await lower_img.read())

      # Run your exact functions
      img_slab = load_image(slab_path)
      img_upper = load_image(upper_path)
      img_lower = load_image(lower_path)

      output_img, info = compose(img_slab, img_upper, img_lower)

      out_path = os.path.join(temp_dir, "output.jpg")
      save_image(output_img, out_path)

      return FileResponse(
          out_path, media_type="image/jpeg", filename="merged.jpg"
      )

  except MergeError as e:
    raise HTTPException(status_code=400, detail=str(e))
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))