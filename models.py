from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

#APPAREL ITEMS

class Category(str, Enum):
    TOPWEAR = "topwear"
    BOTTOMWEAR = "bottomwear"
    OUTERWEAR = "outerwear"
    ONE_PIECE = "one_piece"
    FOOTWEAR = "footwear"
    ACCESSORY = "accessory"

class ApparelItem(BaseModel):
    category: Category = Field(
        description="High-level garment category."
    )
    item_type: str = Field(
        description="Specific garment type, e.g., 'pleated midi skirt', 'cargo pants', 'chelsea boots', 'graphic hoodie'."
    )
    # --- Added for Search & Indexing ---
    title: str = Field(
        description="Canonical e-commerce title: [Brand] [Subject/Graphic if any] [Key Detail] [Item Type]."
    )
    search_query: str = Field(
        description="Targeted 4-7 keyword query optimized for e-commerce search engines."
    )
    brand: Optional[str] = Field(
        default=None,
        description="Visible or recognizable brand/maker (e.g. 'Levi\\'s', 'Adidas', 'Nike'). None if unbranded or uncertain.",
    )
    subject_or_graphic: Optional[str] = Field(
        default=None,
        description="Depicted artwork, sports team/federation, band, character, or prominent text/slogan. None if plain.",
    )
    # -----------------------------------
    primary_color: str = Field(description="Dominant color")
    secondary_colors: list[str] = Field(default_factory=list)
    material_guess: Optional[str] = Field(
        default=None,
        description="Observed fabric/texture, e.g., denim, corduroy, leather, silk, ribbed knit.",
    )
    pattern: Optional[str] = Field(
        default=None,
        description="Visual pattern: solid, striped, plaid, floral, houndstooth, etc.",
    )
    fit_style: Optional[str] = Field(
        default=None,
        description="Silhouettes such as cropped, oversized, slim, flared, high-waisted.",
    )
    distinguishing_features: list[str] = Field(
        default_factory=list,
        description="Specific visible details: zips, patches, distressing, logo placements, cuff style.",
    )

    def __str__(self) -> str:
        descriptors = [
            self.brand,
            self.fit_style.title() if self.fit_style else None,
            self.pattern.title() if self.pattern and self.pattern.lower() != "solid" else None,
            self.primary_color.lower(),
            self.material_guess.lower() if self.material_guess else None,
            self.item_type.lower(),
        ]

        main_phrase = " ".join(word for word in descriptors if word)

        details = []
        if self.subject_or_graphic:
            details.append(f"graphic: {self.subject_or_graphic}")
        if self.secondary_colors:
            details.append(f"accents: {', '.join(self.secondary_colors)}")
        if self.distinguishing_features:
            details.append(f"details: {', '.join(self.distinguishing_features)}")

        details_suffix = f" ({'; '.join(details)})" if details else ""

        return f"[{self.category.value}] {main_phrase}{details_suffix}"

class OutfitAnalysis(BaseModel):
    overall_aesthetic: str = Field(
        description="General style vibe: e.g., streetwear, business casual, avant-garde, athleisure."
    )
    garments: list[ApparelItem]