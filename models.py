from typing import Literal
from datetime import datetime
from enum import Enum
from typing import Optional
from urllib.parse import urlparse
from pydantic import BaseModel, Field, HttpUrl, computed_field


class Category(str, Enum):
    TOPWEAR = "topwear"
    BOTTOMWEAR = "bottomwear"
    OUTERWEAR = "outerwear"
    ONE_PIECE = "one_piece"
    FOOTWEAR = "footwear"
    ACCESSORY = "accessory"


class ApparelFeatures(BaseModel):
    category: Category = Field(description="High-level garment category.")
    item_type: str = Field(description="Specific garment type.")
    title: str = Field(description="Canonical e-commerce title.")
    search_query: str = Field(description="4-7 targeted search keywords.")
    brand: Optional[str] = Field(default=None)
    subject_or_graphic: Optional[str] = Field(default=None)
    primary_color: str = Field(description="Dominant color.")
    secondary_colors: list[str] = Field(default_factory=list)
    material_guess: Optional[str] = Field(default=None)
    pattern: Optional[str] = Field(default=None)
    fit_style: Optional[str] = Field(default=None)
    distinguishing_features: list[str] = Field(default_factory=list)
    overall_aesthetic: Optional[str] = Field(default=None)
    box_2d: list[int] = Field(
        description="Bounding box coordinates for this garment formatted as [ymin, xmin, ymax, xmax] on a 0-1000 normalized integer scale."
    )
    visual_prominence: Literal["primary", "secondary", "background_prop"] = Field(
        description=(
            "Classify the role of this item in the composition: "
            "'primary' if it is the central hero garment being showcased, "
            "'secondary' if it is a supporting styling piece (e.g. beanie, sunglasses, socks, belt), "
            "'background_prop' if it is barely visible."
        )
    )


class CatalogProduct(ApparelFeatures):
    product_url: HttpUrl = Field(description="URL to the product page.")
    image_url: HttpUrl = Field(description="Direct URL to the product image.")
    source_title: Optional[str] = Field(default=None, description="Original raw title from the source platform.")

    @computed_field
    @property
    def seller_domain(self) -> str:
        """Derived automatically from product_url without storing redundant state."""
        return urlparse(str(self.product_url)).netloc


class ApparelSearchQuery(BaseModel):
    raw_query_text: Optional[str] = None
    query_image_path: Optional[str] = None
    inferred_features: Optional[ApparelFeatures] = None
    
    # Metadata filters
    filter_category: Optional[Category] = None
    filter_seller_domain: Optional[str] = None
    filter_brand: Optional[str] = None