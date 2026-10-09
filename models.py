from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional
from urllib.parse import urlparse
from pydantic import BaseModel, Field, HttpUrl, computed_field


# ==========================================
# 1. SHARED ENUMS
# ==========================================

class Category(str, Enum):
    TOPWEAR = "topwear"
    BOTTOMWEAR = "bottomwear"
    OUTERWEAR = "outerwear"
    ONE_PIECE = "one_piece"
    FOOTWEAR = "footwear"
    ACCESSORY = "accessory"


# =================================================================
# FUNCTION 1: DEEP 1-TO-1 APPAREL FEATURE PROFILER
# Use when inspecting a single item or cropped garment in detail.
# =================================================================

class ApparelFeatures(BaseModel):
    """Detailed visual and stylistic profile of a single garment."""
    category: Category = Field(description="High-level garment category.")
    item_type: str = Field(description="Specific garment type (e.g., 'chore jacket', 'cargo pants', 'derby shoes').")
    brand: Optional[str] = Field(default=None, description="Identifiable brand name or visible logo.")
    
    # Colors & Textures
    primary_color: str = Field(description="Dominant color.")
    secondary_colors: list[str] = Field(default_factory=list, description="Accent or trim colors.")
    pattern: Optional[str] = Field(default=None, description="Solid, graphic, striped, houndstooth, floral, etc.")
    material_guess: Optional[str] = Field(default=None, description="Visual fabric texture (denim, corduroy, wool, silk, etc.).")
    
    # Silhouette & Style
    fit_style: Optional[str] = Field(default=None, description="Cut/silhouette (oversized, slim, boxy, cropped, flared).")
    subject_or_graphic: Optional[str] = Field(default=None, description="Description of print, graphic, or embroidery motif.")
    distinguishing_features: list[str] = Field(default_factory=list, description="Pockets, hardware, raw hems, contrast stitching.")
    overall_aesthetic: Optional[str] = Field(default=None, description="Streetwear, minimalist, techwear, gorpcore, vintage.")
    
    # Search Formulation
    search_query: str = Field(description="4-7 concise keywords optimized for catalog retrieval.")


# ==========================================
# 3. BASE APPAREL MODEL
# ==========================================

class ApparelBase(BaseModel):
    """Core domain representation shared by catalog items and search queries."""
    title: Optional[str] = Field(default=None, description="Canonical or working item title.")
    category: Optional[Category] = Field(default=None, description="High-level garment category.")
    item_type: Optional[str] = Field(default=None, description="Granular garment type (e.g., 'bomber jacket', 'cargo pants').")
    features: Optional[ApparelFeatures] = Field(default=None, description="Extracted visual and stylistic features.")


# ==========================================
# 4. DATABASE MODEL (CATALOG STORAGE)
# ==========================================

class CatalogProduct(ApparelBase):
    """Database model stored in MongoDB (or equivalent catalog store).
    Inherits core apparel traits and adds seller information, image links,
    pricing, bulleted features, and vector embedding placeholders.
    """

    product_url: HttpUrl = Field(description="Direct URL to product checkout/detail page.")
    image_url: HttpUrl = Field(description="Direct CDN/storage URL to primary product photo.")
    source_title: Optional[str] = Field(default=None, description="Original uncleaned platform title.")
    

    price: float = Field(ge=0.0, description="Product price.")
    currency: str = Field(default="USD", max_length=3, description="ISO-4217 currency code.")

    key_features: list[str] = Field(default_factory=list, description="Bullet-point product highlights or specs.")


    text_embedding: Optional[list[float]] = Field(default=None, description="512-d CLIP text vector.")
    image_embedding: Optional[list[float]] = Field(default=None, description="512-d CLIP vision vector.")


    @computed_field
    @property
    def seller_domain(self) -> str:
        """Derived dynamically from product_url without redundant state."""
        return urlparse(str(self.product_url)).netloc


# ==========================================
# 5. QUERY MODEL (SEARCH & RETRIEVAL)
# ==========================================

class ApparelQuery(ApparelBase):
    """Search request model for text-based, reverse-image, or hybrid retrieval.
    
    Inherits category/features from ApparelBase to support structured filters,
    while adding raw inputs, vectors, and price bounds.
    """

    raw_query_text: Optional[str] = Field(default=None, description="User search bar text or prompt.")
    query_image_url: Optional[HttpUrl] = Field(default=None, description="URL of an uploaded search image.")
    query_image_path: Optional[str] = Field(default=None, description="Local path to uploaded image buffer.")


    query_vector: Optional[list[float]] = Field(default=None, description="Generated vector for ANN lookup.")


    filter_seller_domains: list[str] = Field(default_factory=list, description="Restrict results to specific domains.")
    filter_brand: Optional[str] = Field(default=None, description="Filter by brand.")
    min_price: Optional[float] = Field(default=None, ge=0.0)
    max_price: Optional[float] = Field(default=None, ge=0.0)
    in_stock_only: bool = Field(default=True)
    top_k: int = Field(default=10, ge=1, le=100)