
"""FastAPI backend for monocular image-to-3D reconstruction."""

import io
import logging
import shutil
import tempfile

from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from uuid import UUID, uuid4

import cv2
import open3d as o3d

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)

from fastapi.responses import FileResponse
from PIL import (
    Image,
    ImageOps,
    UnidentifiedImageError,
)

from src.inference import (
    initialize_model,
    predict_depth,
)

from src.depth_mesh import create_depth_mesh

from src.run_pipeline import (
    generate_foreground_mask,
    make_depth_preview,
)


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "api"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000

DEFAULT_FOV = 60.0
DEFAULT_STRIDE = 1
DEFAULT_BORDER_CROP = 12
DEFAULT_MIN_DEPTH = 0.1
DEFAULT_MAX_RELATIVE_JUMP = 0.03
DEFAULT_SMOOTH_ITERATIONS = 5


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the trained model once when the API starts.

    Reuse it for subsequent reconstruction requests.
    """

    logger.info("Loading ImageTo3D model on CPU...")

    app.state.model = initialize_model()

    # Allow only one CPU reconstruction at a time.
    app.state.reconstruction_lock = Lock()

    logger.info("ImageTo3D model loaded successfully.")

    yield

    # Release references when the server shuts down.
    app.state.model = None


app = FastAPI(
    title="ImageTo3D API",
    description=(
        "Monocular depth estimation and visible-surface "
        "triangle-mesh reconstruction."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/api/health")
def health():
    """Simple API health check."""

    return {
        "status": "ok",
        "inference_device": "cpu",
    }


def decode_uploaded_image(upload: UploadFile):
    """
    Validate an uploaded JPG/PNG image.

    Return a PIL RGB image with EXIF orientation corrected.
    """

    content = upload.file.read(MAX_UPLOAD_BYTES + 1)

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB upload limit.",
        )

    try:
        with Image.open(io.BytesIO(content)) as image:

            if image.format not in ("JPEG", "PNG"):
                raise HTTPException(
                    status_code=415,
                    detail="Only JPEG and PNG images are supported.",
                )

            width, height = image.size

            if (
                width <= 0
                or height <= 0
                or width * height > MAX_IMAGE_PIXELS
            ):
                raise HTTPException(
                    status_code=413,
                    detail="Image exceeds the 16-megapixel limit.",
                )

            image.load()

            # Correct orientation from phone-camera metadata.
            corrected = ImageOps.exif_transpose(image)

            return corrected.convert("RGB")

    except HTTPException:
        raise

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=415,
            detail="The file is not a valid supported image.",
        ) from exc


@app.post(
    "/api/reconstructions",
    status_code=status.HTTP_201_CREATED,
)
def create_reconstruction(
    request: Request,
    file: UploadFile = File(...),
    include_background: bool = Form(False),
):
    """
    Process one image and return URLs for both output files.
    """

    lock = request.app.state.reconstruction_lock

    # Prevent concurrent CPU-intensive reconstructions.
    if not lock.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="Reconstruction server is busy. Retry shortly.",
            headers={"Retry-After": "5"},
        )

    job_dir = None

    try:
        # STEP 1: Validate and decode the uploaded image.
        rgb_image = decode_uploaded_image(file)

        # STEP 2: Create a unique reconstruction identifier.
        job_id = uuid4()

        job_dir = OUTPUT_ROOT / str(job_id)

        # STEP 3: Obtain the loaded model from application state.
        backbone, head, device = request.app.state.model

        # Temporary source file is deleted automatically.
        with tempfile.TemporaryDirectory(
            prefix="imageto3d_"
        ) as temporary_directory:

            input_path = (
                Path(temporary_directory) / "input.jpg"
            )

            # Normalize the uploaded image to JPEG.
            rgb_image.save(
                input_path,
                format="JPEG",
                quality=95,
            )

            # STEP 4: Generate numerical depth.
            depth, rgb = predict_depth(
                input_path,
                backbone,
                head,
                device,
            )

        # STEP 5: Automatic foreground isolation.
        if include_background:
            foreground_mask = None
        else:
            foreground_mask = generate_foreground_mask(
                rgb
            )

        # STEP 6: Generate the triangle mesh.
        mesh, mesh_stats = create_depth_mesh(
            depth,
            rgb,
            horizontal_fov=DEFAULT_FOV,
            stride=DEFAULT_STRIDE,
            border_crop=DEFAULT_BORDER_CROP,
            min_depth=DEFAULT_MIN_DEPTH,
            max_relative_jump=DEFAULT_MAX_RELATIVE_JUMP,
            foreground_mask=foreground_mask,
        )

        # STEP 7: Smooth the mesh.
        mesh = mesh.filter_smooth_taubin(
            number_of_iterations=DEFAULT_SMOOTH_ITERATIONS,
            filter_scope=o3d.geometry.FilterScope.Vertex,
        )

        mesh.compute_vertex_normals()

        # STEP 8: Generate the downloadable depth preview.
        preview = make_depth_preview(
            depth,
            foreground_mask,
        )

        # STEP 9: Save the final output files.
        job_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        depth_path = job_dir / "depth_map.png"
        mesh_path = job_dir / "model.ply"

        if not cv2.imwrite(
            str(depth_path),
            preview,
        ):
            raise OSError("Could not save depth map.")

        if not o3d.io.write_triangle_mesh(
            str(mesh_path),
            mesh,
            write_vertex_colors=True,
            write_vertex_normals=True,
        ):
            raise OSError("Could not save triangle mesh.")

        # STEP 10: Return links to the client.
        base_url = (
            f"/api/reconstructions/{job_id}"
        )

        return {
            "id": str(job_id),
            "status": "completed",
            "device": str(device),
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.triangles),
            "triangles_rejected": (
                mesh_stats["triangles_rejected"]
            ),
            "depth_map_url": f"{base_url}/depth-map",
            "mesh_url": f"{base_url}/mesh",
        }

    except HTTPException:
        raise

    except ValueError as exc:
        logger.warning(
            "Invalid reconstruction input: %s",
            exc,
        )

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("ImageTo3D reconstruction failed.")

        raise HTTPException(
            status_code=500,
            detail=(
                "Reconstruction failed. "
                "Check the server logs."
            ),
        ) from exc

    finally:
        # Remove any incomplete job outputs after an error.
        if job_dir is not None:
            if not (
                (job_dir / "depth_map.png").is_file()
                and (job_dir / "model.ply").is_file()
            ):
                shutil.rmtree(
                    job_dir,
                    ignore_errors=True,
                )

        lock.release()


def get_output_file(job_id: UUID, filename: str):
    """Safely locate a generated reconstruction file."""

    # UUID parsing prevents arbitrary directory names.
    file_path = (
        OUTPUT_ROOT / str(job_id) / filename
    )

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Requested reconstruction file not found.",
        )

    return file_path


@app.get("/api/reconstructions/{job_id}/depth-map")
def download_depth_map(job_id: UUID):
    """Download the generated PNG depth map."""

    path = get_output_file(
        job_id,
        "depth_map.png",
    )

    return FileResponse(
        path=path,
        media_type="image/png",
        filename="depth_map.png",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/api/reconstructions/{job_id}/mesh")
def download_mesh(job_id: UUID):
    """Download the reconstructed PLY triangle mesh."""

    path = get_output_file(
        job_id,
        "model.ply",
    )

    return FileResponse(
        path=path,
        media_type="application/octet-stream",
        filename="model.ply",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )
