#!/usr/bin/env python3
"""
Advanced 3D Novel View Synthesis App
Supports multiple state-of-the-art models with enhanced preprocessing.

IMPORTANT: Run this with the project's virtual environment:
    source venv/bin/activate && python app.py
"""

# CRITICAL: Set environment variables BEFORE any imports to fix Mac M-series crashes
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "CPUExecutionProvider"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import tempfile
import math

# Check if running in correct environment
script_dir = os.path.dirname(os.path.abspath(__file__))
venv_path = os.path.join(script_dir, "venv")
if os.path.exists(venv_path) and "venv" not in sys.executable:
    print("=" * 60)
    print("WARNING: Not running in project virtual environment!")
    print("Please run:")
    print(f"  cd {script_dir}")
    print("  source venv/bin/activate")
    print("  python app.py")
    print("=" * 60)
    # Continue anyway but warn user

sys.path.insert(0, script_dir)

import io
import traceback
import logging
from datetime import datetime

import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
import gradio as gr
import imageio

# ============== LOGGING SETUP ==============

# Create logs directory
LOG_DIR = os.path.join(script_dir, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Setup logging with both file and console output
log_filename = os.path.join(LOG_DIR, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def log_error(context: str, error: Exception, extra_info: dict = None):
    """Log detailed error information for debugging."""
    error_details = {
        "context": context,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "device": DEVICE if 'DEVICE' in globals() else "unknown",
    }
    if extra_info:
        error_details.update(extra_info)

    logger.error("=" * 60)
    logger.error(f"ERROR in {context}")
    logger.error("=" * 60)
    for key, value in error_details.items():
        if key == "traceback":
            logger.error(f"{key}:\n{value}")
        else:
            logger.error(f"{key}: {value}")
    logger.error("=" * 60)

    # Also write to a separate error log for easy access
    error_log_path = os.path.join(LOG_DIR, "errors.log")
    with open(error_log_path, "a") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"ERROR at {error_details['timestamp']}\n")
        f.write(f"Context: {context}\n")
        f.write(f"Type: {error_details['error_type']}\n")
        f.write(f"Message: {error_details['error_message']}\n")
        if extra_info:
            f.write(f"Extra Info: {extra_info}\n")
        f.write(f"Traceback:\n{error_details['traceback']}\n")
        f.write("=" * 60 + "\n")

    return error_details

logger.info(f"Logging initialized. Log file: {log_filename}")

# Lazy load rembg to avoid immediate onnxruntime initialization
REMBG_AVAILABLE = False
REMBG_SESSION = None

def get_rembg_session():
    """Lazily initialize rembg session with CPU provider for stability on Mac."""
    global REMBG_AVAILABLE, REMBG_SESSION
    if REMBG_SESSION is not None:
        logger.debug("Returning cached rembg session")
        return REMBG_SESSION
    try:
        logger.info("Initializing rembg session...")
        from rembg import new_session
        # Force CPU provider to avoid MPS/CoreML crashes on Mac
        # Use isnet-general-use which is smaller and more stable
        REMBG_SESSION = new_session(
            "isnet-general-use",
            providers=["CPUExecutionProvider"]
        )
        REMBG_AVAILABLE = True
        logger.info("rembg initialized successfully with CPU provider")
        return REMBG_SESSION
    except Exception as e:
        log_error("get_rembg_session", e, {"model": "isnet-general-use"})
        REMBG_AVAILABLE = False
        return None

# Determine device
if torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float32
elif torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

logger.info(f"Device configured: {DEVICE}, dtype: {DTYPE}")

# Global pipeline cache
PIPELINES = {}
CURRENT_MODEL = None

# Available models configuration
MODELS = {
    "Stable Zero123": {
        "id": "kxic/stable-zero123",
        "type": "zero123",
        "description": "Original Zero123 - Good quality, fast"
    },
    "Zero123++ v1.2": {
        "id": "sudo-ai/zero123plus-v1.2",
        "type": "zero123plus",
        "description": "Enhanced Zero123 - Better multi-view consistency"
    },
}


def load_pipeline(model_name: str):
    """Load the selected model pipeline."""
    global PIPELINES, CURRENT_MODEL

    logger.info(f"load_pipeline called with model_name={model_name}")

    if model_name in PIPELINES:
        logger.debug(f"Returning cached pipeline for {model_name}")
        CURRENT_MODEL = model_name
        return PIPELINES[model_name]

    if model_name not in MODELS:
        logger.error(f"Unknown model requested: {model_name}")
        return None

    model_config = MODELS[model_name]
    model_id = model_config["id"]
    model_type = model_config["type"]

    logger.info(f"Loading {model_name} pipeline (id={model_id}, type={model_type})...")

    try:
        if model_type == "zero123":
            from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
            from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
            from pipeline_zero1to3 import Zero1to3StableDiffusionPipeline, CCProjection

            # Load on CPU first to avoid MPS crashes, then move to device
            load_dtype = torch.float32  # Always load as float32 for stability

            logger.info("  Loading VAE...")
            vae = AutoencoderKL.from_pretrained(
                model_id, subfolder="vae", torch_dtype=load_dtype,
                low_cpu_mem_usage=True
            )
            logger.debug("  VAE loaded successfully")

            logger.info("  Loading image encoder...")
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                model_id, subfolder="image_encoder", torch_dtype=load_dtype,
                low_cpu_mem_usage=True
            )
            logger.debug("  Image encoder loaded successfully")

            logger.info("  Loading feature extractor...")
            feature_extractor = CLIPImageProcessor.from_pretrained(
                model_id, subfolder="feature_extractor"
            )
            logger.debug("  Feature extractor loaded successfully")

            logger.info("  Loading UNet...")
            unet = UNet2DConditionModel.from_pretrained(
                model_id, subfolder="unet", torch_dtype=load_dtype,
                low_cpu_mem_usage=True
            )
            logger.debug("  UNet loaded successfully")

            logger.info("  Loading scheduler...")
            scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
            logger.debug("  Scheduler loaded successfully")

            logger.info("  Loading CC projection...")
            cc_projection = CCProjection.from_pretrained(model_id, subfolder="cc_projection")
            logger.debug("  CC projection loaded successfully")

            logger.info("  Assembling pipeline...")
            pipeline = Zero1to3StableDiffusionPipeline(
                vae=vae,
                image_encoder=image_encoder,
                unet=unet,
                scheduler=scheduler,
                safety_checker=None,
                feature_extractor=feature_extractor,
                cc_projection=cc_projection,
                requires_safety_checker=False,
            )
            logger.debug("  Pipeline assembled successfully")

            # Move to device after assembly
            logger.info(f"  Moving to {DEVICE}...")
            pipeline = pipeline.to(DEVICE)
            pipeline.enable_attention_slicing()
            logger.debug(f"  Pipeline moved to {DEVICE} with attention slicing enabled")

        elif model_type == "zero123plus":
            from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

            logger.info("  Loading Zero123++ pipeline...")
            pipeline = DiffusionPipeline.from_pretrained(
                model_id,
                custom_pipeline="sudo-ai/zero123plus-pipeline",
                torch_dtype=torch.float32,  # Always float32 for stability
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            logger.debug("  Zero123++ base pipeline loaded")

            pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                pipeline.scheduler.config, timestep_spacing='trailing'
            )
            logger.debug("  Scheduler configured")

            logger.info(f"  Moving to {DEVICE}...")
            pipeline = pipeline.to(DEVICE)
            if hasattr(pipeline, 'enable_attention_slicing'):
                pipeline.enable_attention_slicing()
            logger.debug(f"  Pipeline moved to {DEVICE}")

        PIPELINES[model_name] = pipeline
        CURRENT_MODEL = model_name
        logger.info(f"{model_name} loaded successfully!")
        return pipeline

    except Exception as e:
        log_error("load_pipeline", e, {
            "model_name": model_name,
            "model_id": model_id,
            "model_type": model_type,
            "device": DEVICE
        })
        return None


# ============== ENHANCED SUBJECT DETECTION ==============

def remove_background_enhanced(image: Image.Image) -> Image.Image:
    """
    Enhanced background removal with crystal-clear edges.
    Model-agnostic - same process for all rotation models.
    Falls back to simple threshold if rembg fails.
    """
    logger.debug(f"remove_background_enhanced called, image mode={image.mode}, size={image.size}")

    # Ensure proper mode
    if image.mode == "L":
        image = image.convert("RGB")
        logger.debug("Converted from L to RGB")
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
        logger.debug(f"Converted to RGB from {image.mode}")

    try:
        session = get_rembg_session()
        if session is not None:
            from rembg import remove as rembg_remove
            logger.info("  Using rembg for background removal...")
            # Use session-based removal with CPU provider (more stable on Mac)
            output = rembg_remove(
                image,
                session=session,
                alpha_matting=False,  # Disable alpha matting - causes crashes on Mac
                post_process_mask=True
            )
            logger.info(f"  Background removal successful, output mode={output.mode}, size={output.size}")
            return output
        else:
            logger.warning("  rembg session is None, falling back to simple method")
    except Exception as e:
        log_error("remove_background_enhanced (rembg)", e, {
            "image_mode": image.mode,
            "image_size": str(image.size)
        })

    # Fallback: simple background removal using edge detection
    logger.info("  Using fallback background removal...")
    return fallback_remove_background(image)


def fallback_remove_background(image: Image.Image) -> Image.Image:
    """
    Simple fallback background removal when rembg fails.
    Uses edge detection and flood fill approach.
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    rgb = image.convert("RGB")

    # Detect edges
    edges = rgb.convert("L").filter(ImageFilter.FIND_EDGES)
    edges = edges.filter(ImageFilter.GaussianBlur(radius=2))
    edges_array = np.array(edges)

    # Create mask from edges (anything inside edges is foreground)
    # Use threshold to create initial mask
    gray = np.array(rgb.convert("L"))

    # Assume corners are background - sample corner colors
    h, w = gray.shape
    corner_samples = [
        gray[0:10, 0:10].mean(),
        gray[0:10, w-10:w].mean(),
        gray[h-10:h, 0:10].mean(),
        gray[h-10:h, w-10:w].mean()
    ]
    bg_value = np.mean(corner_samples)

    # Create mask: pixels significantly different from background
    diff = np.abs(gray.astype(float) - bg_value)
    mask = (diff > 30).astype(np.uint8) * 255

    # Add edge information
    mask = np.maximum(mask, edges_array)

    # Clean up mask with morphological operations
    from PIL import ImageFilter
    mask_img = Image.fromarray(mask)
    mask_img = mask_img.filter(ImageFilter.MaxFilter(5))
    mask_img = mask_img.filter(ImageFilter.MinFilter(3))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=1))

    # Apply mask
    result = image.copy()
    result.putalpha(mask_img)

    return result


def get_subject_mask(image: Image.Image) -> Image.Image:
    """Extract clean subject mask from RGBA image."""
    if image.mode != "RGBA":
        return None
    return image.split()[3]


def refine_mask_edges(mask: Image.Image, feather: int = 2) -> Image.Image:
    """Refine mask edges for cleaner cutouts."""
    # Slight blur to smooth jagged edges
    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather * 0.5))

    # Re-threshold to maintain sharp but smooth edges
    mask = mask.point(lambda x: 0 if x < 128 else 255)

    return mask


# ============== DEPTH & BALANCE ANALYSIS ==============

def analyze_image_balance(image: Image.Image) -> dict:
    """
    Analyze image for depth distribution and balance.
    Returns metrics about density clustering and suggested corrections.
    """
    if image.mode == "RGBA":
        # Use alpha to find subject bounds
        alpha = np.array(image.split()[3])
        rgb = np.array(image.convert("RGB"))
    else:
        rgb = np.array(image)
        alpha = np.ones((rgb.shape[0], rgb.shape[1])) * 255

    # Find subject pixels
    subject_mask = alpha > 128
    if not subject_mask.any():
        return {"balanced": True, "correction_angle": 0, "density_shift": 0}

    # Get subject bounds
    rows = np.any(subject_mask, axis=1)
    cols = np.any(subject_mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Calculate center of mass
    y_coords, x_coords = np.where(subject_mask)
    center_y = np.mean(y_coords)
    center_x = np.mean(x_coords)

    # Calculate geometric center
    geo_center_y = (y_min + y_max) / 2
    geo_center_x = (x_min + x_max) / 2

    # Calculate density distribution (top vs bottom)
    mid_y = (y_min + y_max) / 2
    top_density = np.sum(subject_mask[:int(mid_y), :])
    bottom_density = np.sum(subject_mask[int(mid_y):, :])
    total_density = top_density + bottom_density

    density_ratio = top_density / max(total_density, 1)
    density_shift = density_ratio - 0.5  # Positive = top-heavy

    # Calculate tilt angle based on center of mass offset
    height = y_max - y_min
    x_offset = center_x - geo_center_x

    # Estimate rotation needed to straighten
    if height > 0:
        tilt_angle = math.degrees(math.atan2(x_offset, height / 2))
    else:
        tilt_angle = 0

    return {
        "balanced": abs(density_shift) < 0.1,
        "correction_angle": -tilt_angle,  # Negate to correct
        "density_shift": density_shift,
        "center_of_mass": (center_x, center_y),
        "geometric_center": (geo_center_x, geo_center_y),
        "bounds": (x_min, y_min, x_max, y_max)
    }


def correct_image_rotation(image: Image.Image, angle: float) -> Image.Image:
    """Rotate image to correct tilt while preserving alpha."""
    if angle == 0:
        return image

    # Rotate with expand to avoid clipping
    rotated = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)

    return rotated


def balance_depth_distribution(image: Image.Image, shift: float) -> Image.Image:
    """
    Adjust image to balance depth distribution.
    Subtle vertical shift to center the visual weight.
    """
    if abs(shift) < 0.05:
        return image

    # Calculate pixel shift based on density imbalance
    height = image.height
    pixel_shift = int(shift * height * 0.1)  # Subtle correction

    if image.mode == "RGBA":
        # Create new canvas and paste with offset
        new_img = Image.new("RGBA", image.size, (0, 0, 0, 0))
        new_img.paste(image, (0, -pixel_shift))
        return new_img
    else:
        new_img = Image.new("RGB", image.size, (255, 255, 255))
        new_img.paste(image, (0, -pixel_shift))
        return new_img


# ============== IMAGE ADJUSTMENTS ==============

def apply_adjustments(
    image: Image.Image,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    sharpness: float = 1.0,
    blur: float = 0.0
) -> Image.Image:
    """Apply user-controlled adjustments to image."""
    # Preserve alpha if present
    if image.mode == "RGBA":
        alpha = image.split()[3]
        rgb = image.convert("RGB")
    else:
        alpha = None
        rgb = image.convert("RGB")

    # Apply adjustments
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(rgb)
        rgb = enhancer.enhance(brightness)

    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(rgb)
        rgb = enhancer.enhance(contrast)

    if saturation != 1.0:
        enhancer = ImageEnhance.Color(rgb)
        rgb = enhancer.enhance(saturation)

    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(rgb)
        rgb = enhancer.enhance(sharpness)

    if blur > 0:
        rgb = rgb.filter(ImageFilter.GaussianBlur(radius=blur))

    # Restore alpha
    if alpha:
        rgb = rgb.convert("RGBA")
        rgb.putalpha(alpha)

    return rgb


# ============== PREPROCESSING ==============

def preprocess_image(
    image: Image.Image,
    size: int = 256,
    input_resolution: int = 512,
    rotation_correction: float = 0,
    auto_balance: bool = True,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    sharpness: float = 1.0,
    blur: float = 0.0
) -> tuple:
    """
    Full preprocessing pipeline.
    Returns (processed_image, mask_preview, balance_info)
    """
    logger.info(f"preprocess_image called: size={size}, input_res={input_resolution}, "
                f"rotation={rotation_correction}, auto_balance={auto_balance}")

    # Ensure image is valid
    if image is None:
        logger.error("preprocess_image: No image provided")
        raise ValueError("No image provided")

    # Ensure we have a PIL Image
    if not isinstance(image, Image.Image):
        try:
            logger.debug(f"Converting to PIL Image from type {type(image)}")
            image = Image.fromarray(image)
        except Exception as e:
            log_error("preprocess_image (conversion)", e, {"input_type": str(type(image))})
            raise ValueError(f"Cannot convert to PIL Image: {e}")

    logger.debug(f"Input image: mode={image.mode}, size={image.size}")

    # Ensure RGB/RGBA mode
    if image.mode not in ("RGB", "RGBA", "L"):
        logger.debug(f"Converting from {image.mode} to RGB")
        image = image.convert("RGB")

    # Early downscale
    w, h = image.size
    if w == 0 or h == 0:
        logger.error(f"Image has zero dimensions: {w}x{h}")
        raise ValueError("Image has zero dimensions")

    if max(w, h) > input_resolution:
        scale = input_resolution / max(w, h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        logger.debug(f"Downscaling from {w}x{h} to {new_w}x{new_h}")
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Enhanced background removal (model-agnostic)
    logger.info("  Removing background...")
    try:
        image_rgba = remove_background_enhanced(image)
    except Exception as e:
        log_error("preprocess_image (bg removal)", e, {"image_size": str(image.size)})
        image_rgba = fallback_remove_background(image)

    # Get and refine mask
    mask = get_subject_mask(image_rgba)
    if mask:
        try:
            mask = refine_mask_edges(mask)
            image_rgba.putalpha(mask)
            logger.debug("Mask refined and applied")
        except Exception as e:
            log_error("preprocess_image (mask refinement)", e)

    # Analyze balance
    try:
        balance_info = analyze_image_balance(image_rgba)
        logger.debug(f"Balance analysis: {balance_info}")
    except Exception as e:
        log_error("preprocess_image (balance analysis)", e)
        balance_info = {"balanced": True, "correction_angle": 0, "density_shift": 0}

    # Apply rotation correction (manual + auto)
    total_rotation = rotation_correction
    if auto_balance and abs(balance_info["correction_angle"]) > 1:
        total_rotation += balance_info["correction_angle"] * 0.5  # Partial auto-correction

    if total_rotation != 0:
        image_rgba = correct_image_rotation(image_rgba, total_rotation)

    # Balance depth distribution
    if auto_balance:
        image_rgba = balance_depth_distribution(image_rgba, balance_info["density_shift"])

    # Apply user adjustments
    image_rgba = apply_adjustments(
        image_rgba, brightness, contrast, saturation, sharpness, blur
    )

    # Create mask preview
    mask_preview = image_rgba.copy()

    # Crop to subject bounds
    bbox = image_rgba.getbbox()
    if bbox:
        image_rgba = image_rgba.crop(bbox)

    # Resize and center on white background
    w, h = image_rgba.size
    scale = min(size / w, size / h) * 0.85
    new_w, new_h = int(w * scale), int(h * scale)
    image_rgba = image_rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)

    result = Image.new("RGB", (size, size), (255, 255, 255))
    paste_x = (size - new_w) // 2
    paste_y = (size - new_h) // 2

    if image_rgba.mode == "RGBA":
        result.paste(image_rgba, (paste_x, paste_y), image_rgba)
    else:
        result.paste(image_rgba, (paste_x, paste_y))

    return result, mask_preview, balance_info


# ============== PREVIEW FUNCTION ==============

def generate_preview(
    image,
    input_resolution,
    rotation_correction,
    auto_balance,
    brightness,
    contrast,
    saturation,
    sharpness,
    blur
):
    """Generate preview of preprocessed image with current settings."""
    logger.info("generate_preview called")

    if image is None:
        logger.debug("No image provided for preview")
        return None, None, "Upload an image to see preview"

    try:
        # Handle different input types
        if isinstance(image, np.ndarray):
            if image.size == 0:
                logger.warning("Empty image array received")
                return None, None, "Empty image array"
            logger.debug(f"Converting numpy array to PIL: shape={image.shape}, dtype={image.dtype}")
            pil_image = Image.fromarray(image.astype(np.uint8))
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            logger.error(f"Unsupported image type: {type(image)}")
            return None, None, f"Unsupported image type: {type(image)}"

        # Validate input values
        input_resolution = max(128, min(2048, int(input_resolution or 512)))
        rotation_correction = max(-180, min(180, float(rotation_correction or 0)))
        brightness = max(0.1, min(3.0, float(brightness or 1.0)))
        contrast = max(0.1, min(3.0, float(contrast or 1.0)))
        saturation = max(0.0, min(3.0, float(saturation or 1.0)))
        sharpness = max(0.0, min(5.0, float(sharpness or 1.0)))
        blur = max(0.0, min(10.0, float(blur or 0.0)))

        logger.debug(f"Preview params: res={input_resolution}, rot={rotation_correction}, "
                    f"brightness={brightness}, contrast={contrast}")

        logger.info("Generating preview...")
        processed, mask_preview, balance_info = preprocess_image(
            pil_image,
            size=256,
            input_resolution=input_resolution,
            rotation_correction=rotation_correction,
            auto_balance=bool(auto_balance),
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            sharpness=sharpness,
            blur=blur
        )

        status = f"Balance: {'OK' if balance_info.get('balanced', True) else 'Adjusted'}"
        status += f" | Density shift: {balance_info.get('density_shift', 0):.2f}"
        status += f" | Suggested rotation: {balance_info.get('correction_angle', 0):.1f}°"

        logger.info("Preview generated successfully")
        return processed, mask_preview, status

    except Exception as e:
        error_info = log_error("generate_preview", e, {
            "image_type": str(type(image)),
            "input_resolution": input_resolution if 'input_resolution' in dir() else "unknown"
        })
        return None, None, f"Preview error: {str(e)}\n\nSee logs for details: {log_filename}"


# ============== ROTATION GENERATION ==============

def generate_rotation_set(
    image: Image.Image,
    model_name: str,
    angle: float = 30,
    num_steps: int = 75,
    guidance: float = 3.0,
    num_frames: int = 3,
    black_white: bool = False,
    input_resolution: int = 512,
    rotation_correction: float = 0,
    auto_balance: bool = True,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    sharpness: float = 1.0,
    blur: float = 0.0
):
    """Generate rotation frames using selected model."""
    logger.info(f"generate_rotation_set called: model={model_name}, angle={angle}, "
                f"steps={num_steps}, frames={num_frames}")

    # Validate inputs
    if image is None:
        logger.error("No image provided to generate_rotation_set")
        raise ValueError("No image provided")

    if model_name not in MODELS:
        logger.error(f"Unknown model requested: {model_name}")
        raise ValueError(f"Unknown model: {model_name}")

    # Sanitize numeric inputs
    angle = max(1, min(90, float(angle or 30)))
    num_steps = max(10, min(200, int(num_steps or 75)))
    guidance = max(0.5, min(20.0, float(guidance or 3.0)))
    num_frames = max(1, min(20, int(num_frames or 3)))
    input_resolution = max(128, min(2048, int(input_resolution or 512)))

    logger.info(f"Using model: {model_name}")
    logger.info(f"Preprocessing image (input res: {input_resolution}px)...")

    processed, _, _ = preprocess_image(
        image,
        size=256,
        input_resolution=input_resolution,
        rotation_correction=float(rotation_correction or 0),
        auto_balance=bool(auto_balance),
        brightness=float(brightness or 1.0),
        contrast=float(contrast or 1.0),
        saturation=float(saturation or 1.0),
        sharpness=float(sharpness or 1.0),
        blur=float(blur or 0.0)
    )

    print("Loading model pipeline...")
    pipe = load_pipeline(model_name)
    if pipe is None:
        raise RuntimeError(f"Failed to load model: {model_name}")

    model_type = MODELS[model_name]["type"]

    results = []

    # Generate angles
    if num_frames == 1:
        angles = [0]
    else:
        angles = [angle - (2 * angle * i / (num_frames - 1)) for i in range(num_frames)]

    if model_type == "zero123":
        for i, az in enumerate(angles):
            if abs(az) < 0.01:
                frame = processed
                print(f"Frame {i+1}/{num_frames}: Using original (azimuth=0°)")
            else:
                print(f"Frame {i+1}/{num_frames}: Generating view at azimuth={az:.1f}°...")
                pose = [0, az, 0.0]

                try:
                    with torch.no_grad():
                        frame = pipe(
                            input_imgs=processed,
                            prompt_imgs=processed,
                            poses=[pose],
                            height=256,
                            width=256,
                            num_inference_steps=num_steps,
                            guidance_scale=guidance,
                        ).images[0]
                except Exception as e:
                    print(f"Frame generation failed: {e}")
                    frame = processed  # Use original as fallback

            if black_white:
                frame = frame.convert("L").convert("RGB")

            results.append(frame)

    elif model_type == "zero123plus":
        # Zero123++ generates 6 views at once
        print("Generating multi-view images with Zero123++...")
        try:
            with torch.no_grad():
                output = pipe(processed, num_inference_steps=num_steps).images[0]

            # Zero123++ outputs a grid - split into individual views
            w, h = output.size
            view_w = w // 3
            view_h = h // 2

            views = []
            for row in range(2):
                for col in range(3):
                    view = output.crop((col * view_w, row * view_h, (col + 1) * view_w, (row + 1) * view_h))
                    views.append(view)

            # Select views based on requested frames
            if num_frames <= len(views):
                step = max(1, len(views) // num_frames)
                results = [views[i * step] for i in range(min(num_frames, len(views)))]
            else:
                results = views

            if black_white:
                results = [f.convert("L").convert("RGB") for f in results]

        except Exception as e:
            print(f"Zero123++ generation failed: {e}")
            # Return copies of processed image as fallback
            results = [processed.copy() for _ in range(num_frames)]

    # Ensure we have at least one result
    if not results:
        results = [processed]

    return results


# ============== GIF CREATION ==============

def create_gif(images: list, duration: float = 0.5, boomerang: bool = False) -> bytes:
    """Create GIF from images with optional boomerang effect."""
    if not images:
        raise ValueError("No images provided for GIF creation")

    try:
        gif_buffer = io.BytesIO()
        frames = []

        for img in images:
            if isinstance(img, Image.Image):
                # Ensure RGB mode
                if img.mode != "RGB":
                    img = img.convert("RGB")
                frames.append(np.array(img))
            elif isinstance(img, np.ndarray):
                frames.append(img)
            else:
                print(f"Skipping invalid frame type: {type(img)}")

        if not frames:
            raise ValueError("No valid frames for GIF")

        if boomerang and len(frames) >= 3:
            mid = len(frames) // 2
            frames = frames[mid:] + frames[mid - 1::-1] + frames[1:mid]

        imageio.mimsave(gif_buffer, frames, format='GIF', duration=duration, loop=0)
        gif_buffer.seek(0)
        return gif_buffer.getvalue()

    except Exception as e:
        print(f"GIF creation error: {e}")
        raise


# ============== EXPORT ==============

def export_images(images: list, output_dir: str, base_name: str = "view"):
    """Export images to directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for i, img in enumerate(images):
        filename = f"{base_name}_frame_{i:02d}.png"
        filepath = output_path / filename
        img.save(filepath)
        saved_paths.append(str(filepath))
        print(f"Saved: {filepath}")

    gif_path = output_path / f"{base_name}_animation.gif"
    gif_data = create_gif(images)
    with open(gif_path, "wb") as f:
        f.write(gif_data)
    saved_paths.append(str(gif_path))
    print(f"Saved: {gif_path}")

    return saved_paths


# ============== GLOBALS ==============

CURRENT_IMAGES = []


# ============== MAIN PROCESS ==============

def process_image(
    image,
    model_name,
    angle,
    num_steps,
    guidance,
    num_frames,
    black_white,
    boomerang,
    input_resolution,
    rotation_correction,
    auto_balance,
    brightness,
    contrast,
    saturation,
    sharpness,
    blur
):
    """Main processing function."""
    global CURRENT_IMAGES

    if image is None:
        return None, None, "Please upload an image first."

    # Validate image array
    if isinstance(image, np.ndarray):
        if image.size == 0:
            return None, None, "Empty image provided."
        if len(image.shape) < 2:
            return None, None, "Invalid image dimensions."

    try:
        # Convert to PIL Image safely
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype(np.uint8))
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            return None, None, f"Unsupported image type: {type(image)}"

        print(f"Processing image: {pil_image.size}, mode={pil_image.mode}")

        images = generate_rotation_set(
            pil_image,
            model_name=model_name or "Stable Zero123",
            angle=float(angle or 30),
            num_steps=int(num_steps or 75),
            guidance=float(guidance or 3.0),
            num_frames=int(num_frames or 3),
            black_white=bool(black_white),
            input_resolution=int(input_resolution or 512),
            rotation_correction=float(rotation_correction or 0),
            auto_balance=bool(auto_balance),
            brightness=float(brightness or 1.0),
            contrast=float(contrast or 1.0),
            saturation=float(saturation or 1.0),
            sharpness=float(sharpness or 1.0),
            blur=float(blur or 0.0)
        )

        if not images:
            return None, None, "No frames generated."

        CURRENT_IMAGES = images

        gif_data = create_gif(images, duration=0.5, boomerang=bool(boomerang))
        gif_path = os.path.join(tempfile.gettempdir(), "preview.gif")
        with open(gif_path, "wb") as f:
            f.write(gif_data)

        return images, gif_path, f"Generated {len(images)} frames with {model_name}!"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"Error: {str(e)}"


def do_export(output_dir):
    """Export current images."""
    global CURRENT_IMAGES

    if not CURRENT_IMAGES:
        return "No images to export. Generate images first."

    if not output_dir:
        return "Please specify an output directory."

    try:
        saved = export_images(CURRENT_IMAGES, output_dir)
        return f"Exported {len(saved)} files to {output_dir}"
    except Exception as e:
        return f"Export error: {str(e)}"


# ============== UI ==============

def get_theme():
    """Get the Gradio theme."""
    return gr.themes.Base(
        primary_hue="blue",
        secondary_hue="gray",
    ).set(
        button_primary_background_fill="#2563eb",
        button_primary_background_fill_hover="#1d4ed8",
        button_primary_text_color="white",
        block_title_text_weight="600",
        block_label_text_weight="500",
    )


def create_ui():
    """Create the Gradio interface."""

    with gr.Blocks(title="Advanced 3D Novel View Synthesis") as app:
        gr.Markdown("# Advanced 3D Novel View Synthesis")
        gr.Markdown("Upload an image to generate true 3D rotated views with enhanced preprocessing.")

        with gr.Row():
            # Left column - Controls
            with gr.Column(scale=1):
                input_image = gr.Image(label="Upload Image", type="numpy")

                # Model Selection
                with gr.Group():
                    gr.Markdown("### Model Selection")
                    model_dropdown = gr.Dropdown(
                        choices=list(MODELS.keys()),
                        value="Stable Zero123",
                        label="3D Model"
                    )
                    model_info = gr.Textbox(
                        value=MODELS["Stable Zero123"]["description"],
                        label="Model Info",
                        interactive=False
                    )

                # Generation Settings
                with gr.Group():
                    gr.Markdown("### Generation Settings")
                    angle_slider = gr.Slider(
                        minimum=10, maximum=60, value=30, step=5,
                        label="Rotation Angle (±degrees)"
                    )
                    frames_slider = gr.Slider(
                        minimum=3, maximum=12, value=3, step=1,
                        label="Number of Frames"
                    )
                    steps_slider = gr.Slider(
                        minimum=25, maximum=100, value=75, step=5,
                        label="Inference Steps"
                    )
                    guidance_slider = gr.Slider(
                        minimum=1.0, maximum=10.0, value=3.0, step=0.5,
                        label="Guidance Scale"
                    )

                # Image Adjustments
                with gr.Group():
                    gr.Markdown("### Image Adjustments")
                    brightness_slider = gr.Slider(
                        minimum=0.5, maximum=1.5, value=1.0, step=0.05,
                        label="Brightness"
                    )
                    contrast_slider = gr.Slider(
                        minimum=0.5, maximum=1.5, value=1.0, step=0.05,
                        label="Contrast"
                    )
                    saturation_slider = gr.Slider(
                        minimum=0.5, maximum=1.5, value=1.0, step=0.05,
                        label="Saturation"
                    )
                    sharpness_slider = gr.Slider(
                        minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                        label="Sharpness"
                    )
                    blur_slider = gr.Slider(
                        minimum=0.0, maximum=2.0, value=0.0, step=0.1,
                        label="Blur"
                    )

                # Advanced Settings
                with gr.Group():
                    gr.Markdown("### Advanced Settings")
                    input_res_slider = gr.Slider(
                        minimum=256, maximum=1024, value=512, step=128,
                        label="Input Resolution"
                    )
                    rotation_correction_slider = gr.Slider(
                        minimum=-45, maximum=45, value=0, step=1,
                        label="Rotation Correction (degrees)"
                    )
                    auto_balance_checkbox = gr.Checkbox(
                        value=True,
                        label="Auto-balance depth distribution"
                    )
                    bw_checkbox = gr.Checkbox(
                        value=False,
                        label="Black & White"
                    )
                    boomerang_checkbox = gr.Checkbox(
                        value=False,
                        label="Boomerang GIF"
                    )

                preview_btn = gr.Button("Update Preview", variant="secondary")
                generate_btn = gr.Button("Generate 3D Views", variant="primary")

                # Export
                with gr.Group():
                    gr.Markdown("### Export")
                    output_dir = gr.Textbox(
                        label="Export Directory",
                        placeholder="/path/to/output/folder"
                    )
                    export_btn = gr.Button("Export Images")

                status = gr.Textbox(label="Status", interactive=False)

            # Right column - Outputs
            with gr.Column(scale=2):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Preprocessed Preview")
                        preview_image = gr.Image(label="Processed Input", type="pil")
                    with gr.Column():
                        gr.Markdown("### Mask Preview")
                        mask_image = gr.Image(label="Subject Mask", type="pil")

                gr.Markdown("### Generated Frames")
                gallery = gr.Gallery(label="Generated Frames", columns=4, height="auto")

                gr.Markdown("### Animation Preview")
                gif_preview = gr.Image(label="GIF Preview", type="filepath")

        # Update model info on selection
        def update_model_info(model_name):
            return MODELS[model_name]["description"]

        model_dropdown.change(
            fn=update_model_info,
            inputs=[model_dropdown],
            outputs=[model_info]
        )

        # Preview controls
        preview_inputs = [
            input_image, input_res_slider, rotation_correction_slider,
            auto_balance_checkbox, brightness_slider, contrast_slider,
            saturation_slider, sharpness_slider, blur_slider
        ]

        preview_btn.click(
            fn=generate_preview,
            inputs=preview_inputs,
            outputs=[preview_image, mask_image, status]
        )

        # Auto-update preview on slider change
        for slider in [brightness_slider, contrast_slider, saturation_slider,
                       sharpness_slider, blur_slider, rotation_correction_slider]:
            slider.release(
                fn=generate_preview,
                inputs=preview_inputs,
                outputs=[preview_image, mask_image, status]
            )

        # Generate
        generate_btn.click(
            fn=process_image,
            inputs=[
                input_image, model_dropdown, angle_slider, steps_slider,
                guidance_slider, frames_slider, bw_checkbox, boomerang_checkbox,
                input_res_slider, rotation_correction_slider, auto_balance_checkbox,
                brightness_slider, contrast_slider, saturation_slider,
                sharpness_slider, blur_slider
            ],
            outputs=[gallery, gif_preview, status]
        )

        # Export
        export_btn.click(
            fn=do_export,
            inputs=[output_dir],
            outputs=[status]
        )

    return app


def main():
    """Main entry point."""
    print("=" * 50)
    print("Advanced 3D Novel View Synthesis")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    print(f"Available models: {list(MODELS.keys())}")
    print("=" * 50)

    print("Starting web UI at http://127.0.0.1:7860")

    app = create_ui()
    app.launch(share=False, server_name="127.0.0.1", server_port=7860, theme=get_theme())


if __name__ == "__main__":
    main()
