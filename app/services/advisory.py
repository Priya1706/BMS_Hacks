"""Disease Advisory Service for RootCause.

This module provides structured advisory and agronomic guidance for crop diseases
affecting rice and wheat crops.

IMPORTANT NOTICE:
All advisory content in this module contains educational placeholder data for
system development and testing. It does NOT provide certified pesticide dosages or
prescriptive chemical recommendations. All recommendations clearly require consultation
with certified local agricultural extension workers before field application.
"""

from typing import Any, Dict, List, Optional

DISCLAIMER_TEXT = (
    "DISCLAIMER: This advisory contains educational placeholder data for prototype testing. "
    "Do not apply chemical treatments or calculate dosages without consulting a certified "
    "local agricultural extension officer or agronomist. Always adhere to national and regional "
    "plant protection guidelines."
)

# Placeholder disease database for rice
RICE_DISEASES: Dict[str, Dict[str, Any]] = {
    "leaf_blast": {
        "disease_id": "leaf_blast",
        "disease_name": "Rice Blast",
        "scientific_name": "Magnaporthe oryzae",
        "crop_type": "rice",
        "is_placeholder": True,
        "severity_level": "High",
        "symptoms": [
            "Spindle-shaped or diamond-shaped spots on leaves with gray or whitish centers and brown margins.",
            "Lesions enlarge and coalesce, causing entire leaves to wither and desiccate.",
            "Neck or collar rot leading to whiteheads or lodging in severe cases.",
        ],
        "preventive_measures": [
            "Use certified blast-resistant or tolerant rice cultivars suitable for your agro-ecological zone.",
            "Avoid excessive use of nitrogenous fertilizers; apply nitrogen in split doses based on leaf color charts (LCC).",
            "Maintain optimal plant spacing to avoid dense canopy humidity that fosters fungal spore germination.",
            "Destroy crop residue and infected stubble after harvest.",
        ],
        "organic_controls": [
            "Apply bio-control agents such as Pseudomonas fluorescens or Trichoderma harzianum as seed treatment or foliar spray (consult local extension recommendations).",
            "Ensure clean field sanitation and weed management to remove alternate grass hosts.",
        ],
        "chemical_guidance": (
            "NOTICE: Chemical fungicides may be necessary under severe epidemic conditions. "
            "Consult your local agricultural extension service for authorized active ingredients, "
            "approved local brands, application timing, and pre-harvest intervals. "
            "Never apply unverified dosages."
        ),
        "monitoring_advice": "Inspect fields every 3 to 5 days during tillering and panicle initiation stages, especially during cloudy, humid weather.",
        "disclaimer": DISCLAIMER_TEXT,
    },
    "bacterial_leaf_blight": {
        "disease_id": "bacterial_leaf_blight",
        "disease_name": "Bacterial Leaf Blight",
        "scientific_name": "Xanthomonas oryzae pv. oryzae",
        "crop_type": "rice",
        "is_placeholder": True,
        "severity_level": "High",
        "symptoms": [
            "Water-soaked lesions beginning at the leaf margins and tips.",
            "Lesions enlarge into undulating yellow-to-white stripes along leaf edges.",
            "Milky bacterial exudate droplets visible on young lesions early in the morning.",
        ],
        "preventive_measures": [
            "Plant resistant crop varieties recommended by the local agriculture department.",
            "Avoid field-to-field irrigation flow from known infected plots.",
            "Ensure balanced fertilization with adequate potassium and avoid excess nitrogen.",
            "Maintain good drainage during the vegetative stage.",
        ],
        "organic_controls": [
            "Practice seed treatment with bio-formulations approved in your district.",
            "Allow fields to dry intermittently to reduce bacterial persistence.",
        ],
        "chemical_guidance": (
            "NOTICE: Bacterial diseases cannot be treated with standard fungal sprays. "
            "Consult regional extension specialists for bactericidal guidelines and district protocols. "
            "No chemical dosage is prescribed here."
        ),
        "monitoring_advice": "Check upper leaf tips in early morning for bacterial ooze and water-soaking after rainstorms or high wind events.",
        "disclaimer": DISCLAIMER_TEXT,
    },
    "brown_spot": {
        "disease_id": "brown_spot",
        "disease_name": "Brown Spot",
        "scientific_name": "Bipolaris oryzae",
        "crop_type": "rice",
        "is_placeholder": True,
        "severity_level": "Medium",
        "symptoms": [
            "Small, oval to circular brown spots on leaf blades with yellow halos.",
            "Spots may coalesce causing leaf yellowing and reduced photosynthetic area.",
            "Discolored, spotted seeds on the panicle.",
        ],
        "preventive_measures": [
            "Correct soil nutrient deficiencies, particularly potassium, manganese, and silicon.",
            "Use certified disease-free seed and practice hot water or bio-agent seed treatment.",
            "Improve soil organic matter and avoid water stress in drought-prone fields.",
        ],
        "organic_controls": [
            "Incorporate neem cake into soil preparation.",
            "Maintain soil health and organic amendments.",
        ],
        "chemical_guidance": (
            "NOTICE: Targeted fungicide seed treatments or protective sprays may be advised by local authorities. "
            "Consult local extension officers for approved products. No dosages are provided."
        ),
        "monitoring_advice": "Monitor stressed or nutrient-deficient field patches regularly during vegetative stages.",
        "disclaimer": DISCLAIMER_TEXT,
    },
    "leaf_scald": {
        "disease_id": "leaf_scald",
        "disease_name": "Leaf Scald",
        "scientific_name": "Microdochium oryzae",
        "crop_type": "rice",
        "is_placeholder": True,
        "severity_level": "Medium",
        "symptoms": [
            "Large lesions starting at leaf tips or leaf margins.",
            "Characteristic chevron or zonate pattern of alternating light brown and dark brown bands.",
            "Infected leaves dry up from the tip down and display a scorched or scalded appearance.",
        ],
        "preventive_measures": [
            "Utilize certified disease-free seeds and resistant cultivars.",
            "Avoid excessive nitrogen fertilization; follow balanced split dosing.",
            "Maintain field sanitation and manage alternate weed hosts along bunds.",
        ],
        "organic_controls": [
            "Seed treatment with approved antagonistic bio-control formulations.",
            "Deep summer plowing to bury infected stubble and sclerotia.",
        ],
        "chemical_guidance": (
            "NOTICE: Chemical interventions should only be made following official pest surveillance advice. "
            "Consult your local agricultural extension service for authorized formulations. No chemical dosages are prescribed."
        ),
        "monitoring_advice": "Scout fields bi-weekly during tillering and panicle development stages.",
        "disclaimer": DISCLAIMER_TEXT,
    },
    "sheath_blight": {
        "disease_id": "sheath_blight",
        "disease_name": "Sheath Blight",
        "scientific_name": "Rhizoctonia solani",
        "crop_type": "rice",
        "is_placeholder": True,
        "severity_level": "High",
        "symptoms": [
            "Oval or elliptical greenish-grey, water-soaked spots on leaf sheaths near the water line.",
            "Lesions enlarge with greyish-white centers and irregular reddish-brown borders.",
            "Lesions coalesce and spread upward to leaf blades, resulting in canopy death and lodging.",
        ],
        "preventive_measures": [
            "Maintain optimal plant density and wider spacing to reduce humidity in the micro-canopy.",
            "Avoid high nitrogen doses; ensure adequate potassium supply.",
            "Practice intermittent drainage (alternate wetting and drying) to lower canopy moisture.",
        ],
        "organic_controls": [
            "Foliar application or soil amendment of Trichoderma harzianum or Pseudomonas fluorescens.",
            "Burn or compost infected stubble post-harvest to suppress sclerotia survival.",
        ],
        "chemical_guidance": (
            "NOTICE: Severe sheath blight may require registered systemic fungicide applications. "
            "Consult local agricultural extension agronomists for certified products and timings. Never apply unverified dosages."
        ),
        "monitoring_advice": "Scout the base of hills near the waterline weekly from maximum tillering onwards.",
        "disclaimer": DISCLAIMER_TEXT,
    },
    "healthy": {
        "disease_id": "healthy",
        "disease_name": "Healthy Rice Plant",
        "scientific_name": "Oryza sativa",
        "crop_type": "rice",
        "is_placeholder": True,
        "severity_level": "None",
        "symptoms": [
            "Uniform green leaf pigmentation without abnormal lesions, chlorosis, or necrotic spots.",
            "Vigorous tillering and healthy root architecture.",
        ],
        "preventive_measures": [
            "Continue standard Integrated Pest and Nutrient Management (IPNM) practices.",
            "Maintain regular field scouting to detect early signs of stress or pest pressure.",
            "Follow standard water management (alternate wetting and drying where applicable).",
        ],
        "organic_controls": [
            "Apply compost and bio-fertilizers to sustain soil microbial diversity.",
        ],
        "chemical_guidance": "No chemical intervention required. Preserve beneficial predator populations.",
        "monitoring_advice": "Maintain weekly field inspections.",
        "disclaimer": DISCLAIMER_TEXT,
    },
}

# Placeholder disease database for wheat
WHEAT_DISEASES: Dict[str, Dict[str, Any]] = {
    "leaf_rust": {
        "disease_id": "leaf_rust",
        "disease_name": "Wheat Leaf Rust (Brown Rust)",
        "scientific_name": "Puccinia triticina",
        "crop_type": "wheat",
        "is_placeholder": True,
        "severity_level": "High",
        "symptoms": [
            "Small, round-to-oval orange-brown pustules scattered randomly on the upper leaf surface.",
            "Dusty orange-brown urediniospores rubbing off on fingers upon contact.",
            "Premature leaf drying leading to shriveled grains.",
        ],
        "preventive_measures": [
            "Cultivate rust-resistant wheat varieties recommended for your wheat growing zone.",
            "Eradicate volunteer wheat plants and alternate hosts between cropping seasons.",
            "Follow recommended sowing dates to avoid peak rust spore dispersal periods.",
        ],
        "organic_controls": [
            "Encourage diverse crop rotations to disrupt the pathogen lifecycle.",
            "Utilize bio-fungicide formulations tested and approved by state agricultural universities.",
        ],
        "chemical_guidance": (
            "NOTICE: Triazole or strobilurin fungicides are sometimes recommended during early rust detection. "
            "Always consult your local Krishi Vigyan Kendra (KVK) or extension agronomist for approved treatments. "
            "No dosages are provided."
        ),
        "monitoring_advice": "Check lower and middle leaves weekly starting from early jointing through flag leaf emergence.",
        "disclaimer": DISCLAIMER_TEXT,
    },
    "stripe_rust": {
        "disease_id": "stripe_rust",
        "disease_name": "Wheat Stripe Rust (Yellow Rust)",
        "scientific_name": "Puccinia striiformis",
        "crop_type": "wheat",
        "is_placeholder": True,
        "severity_level": "Very High",
        "symptoms": [
            "Bright yellow pustules arranged in distinct narrow stripes along the leaf veins.",
            "Pustules rupture releasing yellow powdery spores.",
            "Leaves turn chlorotic and necrotic, causing significant yield reduction.",
        ],
        "preventive_measures": [
            "Adopt certified yellow-rust resistant cultivars adapted to cooler agro-climatic conditions.",
            "Avoid late sowing in yellow-rust prone northern belts.",
            "Scout fields early, especially in low-lying and shaded areas.",
        ],
        "organic_controls": [
            "Implement community-wide resistant variety adoption to lower regional spore pressure.",
        ],
        "chemical_guidance": (
            "NOTICE: Immediate localized spray may be advised if yellow rust crosses economic threshold levels. "
            "Contact your district agricultural officer immediately. No dosage is provided here."
        ),
        "monitoring_advice": "Scout fields twice a week during cool, damp weather conditions (10-15°C).",
        "disclaimer": DISCLAIMER_TEXT,
    },
    "powdery_mildew": {
        "disease_id": "powdery_mildew",
        "disease_name": "Wheat Powdery Mildew",
        "scientific_name": "Blumeria graminis f. sp. tritici",
        "crop_type": "wheat",
        "is_placeholder": True,
        "severity_level": "Medium",
        "symptoms": [
            "White, fluffy, powdery fungal patches on lower leaves, stems, and leaf sheaths.",
            "Patches turn dull gray or brown with tiny black fruiting bodies (cleistothecia) over time.",
            "Lower leaves turn yellow and die prematurely.",
        ],
        "preventive_measures": [
            "Avoid overly dense plant stands to improve air circulation within the crop canopy.",
            "Avoid excessive nitrogen applications that promote soft, lush foliage.",
            "Practice deep plowing of infected crop residue.",
        ],
        "organic_controls": [
            "Foliar bio-control sprays approved for organic grain production.",
        ],
        "chemical_guidance": (
            "NOTICE: Consult your local agricultural officer for region-specific chemical advisories. "
            "No chemical dosages are specified."
        ),
        "monitoring_advice": "Scout lower leaf canopy in dense stands during high relative humidity.",
        "disclaimer": DISCLAIMER_TEXT,
    },
    "healthy": {
        "disease_id": "healthy",
        "disease_name": "Healthy Wheat Plant",
        "scientific_name": "Triticum aestivum",
        "crop_type": "wheat",
        "is_placeholder": True,
        "severity_level": "None",
        "symptoms": [
            "Clean, dark green foliage without visible rust pustules, powdery mildew, or necrotic streaks.",
            "Sturdy tillers and healthy flag leaves.",
        ],
        "preventive_measures": [
            "Maintain timely irrigation at critical growth stages (CRI, tillering, flowering).",
            "Follow balanced N:P:K fertilization guidelines based on soil test results.",
            "Continue periodic scouting through grain filling.",
        ],
        "organic_controls": [
            "Maintain soil organic carbon with farmyard manure and green manuring.",
        ],
        "chemical_guidance": "No chemical applications warranted.",
        "monitoring_advice": "Scout weekly until harvest.",
        "disclaimer": DISCLAIMER_TEXT,
    },
    "black_rust": {
        "disease_id": "black_rust",
        "disease_name": "Wheat Stem Rust (Black Rust)",
        "scientific_name": "Puccinia graminis f. sp. tritici",
        "crop_type": "wheat",
        "is_placeholder": True,
        "severity_level": "Very High",
        "symptoms": [
            "Elongated reddish-brown to black pustules on stems, leaf sheaths, and glumes.",
            "The outer stem tissue tears around the pustules, leaving ragged edges.",
            "Late pustules darken as black teliospores form, and weakened stems may lodge.",
        ],
        "preventive_measures": [
            "Grow stem-rust resistant varieties recommended for your wheat zone.",
            "Remove volunteer wheat and nearby alternate hosts such as barberry.",
            "Avoid very late sowing in warmer central and southern zones.",
        ],
        "organic_controls": [
            "Maintain field sanitation and crop rotation to reduce local inoculum.",
            "Use bio-fungicide formulations only when approved by the local extension service.",
        ],
        "chemical_guidance": (
            "NOTICE: Stem rust can spread quickly in warm, dewy weather. "
            "Consult your local Krishi Vigyan Kendra or extension agronomist before any spray. "
            "No dosage is provided here."
        ),
        "monitoring_advice": "Inspect stems and sheaths twice a week once temperatures rise above 20°C.",
        "disclaimer": DISCLAIMER_TEXT,
    },
}

ALL_DISEASES = {
    "rice": RICE_DISEASES,
    "wheat": WHEAT_DISEASES,
}


def get_disease_advisory(crop_type: str, disease_id: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve structured advisory information for a given crop disease.

    All content returned is placeholder data clearly marked for prototyping.

    Args:
        crop_type: Target crop ('rice' or 'wheat').
        disease_id: Optional specific disease key (e.g. 'leaf_blast'). If None or
                    unrecognized, returns general crop management guidance.

    Returns:
        dict: Structured advisory containing symptoms, prevention, organic control,
              safe chemical guidance notice, and legal disclaimer.
    """
    normalized_crop = crop_type.strip().lower() if crop_type else ""

    if normalized_crop not in ALL_DISEASES:
        return {
            "is_placeholder": True,
            "crop_type": crop_type,
            "disease_name": "Unknown Crop",
            "message": f"Unsupported crop '{crop_type}'. Currently supported crops are 'rice' and 'wheat'.",
            "disclaimer": DISCLAIMER_TEXT,
        }

    crop_dict = ALL_DISEASES[normalized_crop]

    if disease_id:
        clean_id = disease_id.strip().lower().replace(" ", "_").replace("-", "_")
        # Display labels returned by the wheat classifier use field names,
        # while existing advisory records use conventional disease IDs.
        clean_id = {
            "brown_rust": "leaf_rust",
            "yellow_rust": "stripe_rust",
            "stem_rust": "black_rust",
        }.get(clean_id, clean_id)
        if clean_id in crop_dict:
            return crop_dict[clean_id].copy()
        if clean_id.startswith("healthy") and "healthy" in crop_dict:
            return crop_dict["healthy"].copy()

    # If no specific disease ID or unknown disease ID, provide general crop management advice
    return {
        "disease_id": disease_id or "general_guidance",
        "disease_name": f"General {normalized_crop.capitalize()} Health Management",
        "crop_type": normalized_crop,
        "is_placeholder": True,
        "severity_level": "Informational",
        "symptoms": [
            "No specific disease identified or model prediction pending.",
        ],
        "preventive_measures": [
            "Use certified seeds treated with recommended bio-fungicides.",
            "Adopt crop rotation and field sanitation to suppress soil-borne pathogens.",
            "Practice balanced fertilizer application according to local soil health card recommendations.",
        ],
        "organic_controls": [
            "Incorporate green manure or well-decomposed organic compost.",
            "Promote biodiversity along field bunds to harbor natural predators.",
        ],
        "chemical_guidance": (
            "NOTICE: Never apply pesticides or fungicides preemptively without confirmed diagnosis "
            "and prescription from certified agricultural extension personnel."
        ),
        "monitoring_advice": "Perform routine weekly field inspections at morning hours.",
        "disclaimer": DISCLAIMER_TEXT,
    }


def list_supported_diseases(crop_type: Optional[str] = None) -> Dict[str, Any]:
    """List all supported diseases and placeholder records by crop.

    Args:
        crop_type: Optional crop filter ('rice' or 'wheat').

    Returns:
        dict: Mapping of crop types to supported disease summaries.
    """
    if crop_type:
        normalized_crop = crop_type.strip().lower()
        if normalized_crop not in ALL_DISEASES:
            return {"error": f"Unsupported crop '{crop_type}'. Supported: 'rice', 'wheat'."}
        return {
            normalized_crop: [
                {
                    "disease_id": d_id,
                    "disease_name": data["disease_name"],
                    "scientific_name": data.get("scientific_name", ""),
                    "is_placeholder": True,
                }
                for d_id, data in ALL_DISEASES[normalized_crop].items()
            ]
        }

    return {
        crop: [
            {
                "disease_id": d_id,
                "disease_name": data["disease_name"],
                "scientific_name": data.get("scientific_name", ""),
                "is_placeholder": True,
            }
            for d_id, data in crop_diseases.items()
        ]
        for crop, crop_diseases in ALL_DISEASES.items()
    }
