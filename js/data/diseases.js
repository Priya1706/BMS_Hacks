/**
 * RootCause Disease Knowledge Base
 * Offline-first comprehensive plant pathology database
 * Supported Crops: Rice (Paddy) and Wheat
 */

export const DISEASES_DATABASE = {
  rice: [
    {
      id: 'rice_bacterial_leaf_blight',
      name: 'Bacterial Leaf Blight',
      localName: 'बैक्टीरियल पत्ती झुलसा / ಬ್ಯಾಕ್ಟೀರಿಯಾದ ಎಲೆ ರೋಗ',
      crop: 'rice',
      cropDisplayName: 'Rice (धान)',
      scientificName: 'Xanthomonas oryzae pv. oryzae',
      severity: 'High',
      season: 'Kharif (Monsoon)',
      priority: 'High Priority',
      summary: 'A devastating bacterial disease that starts with water-soaked stripes at leaf margins, turning greyish-white and causing severe drying of leaves and poor grain filling.',
      symptoms: [
        'Water-soaked lesions begin at leaf margins near tip.',
        'Lesions enlarge along edges with wavy margins, turning yellowish-white to grey.',
        'Milky or opaque bacterial ooze droplets appear on young lesions early morning.',
        'Severely infected leaves wilt and roll up (known as "kresek" in early stages).'
      ],
      visibleSigns: 'Yellow to straw-colored wavy striping starting from leaf tips and margins; dries out to papery grey.',
      causes: 'Bacterial pathogen entering through leaf wounds or natural pores. Spreads rapidly in high humidity (80%+), heavy monsoon rainfall, warm temperatures (28-34°C), and excessive nitrogen fertilization.',
      treatment: {
        organic: [
          'Spray fresh Cow Dung Extract (20g fresh cow dung stirred in 1 liter water, filtered) at 10-day intervals.',
          'Apply Pseudomonas fluorescens (talc formulation) @ 5g/liter of water as foliar spray.',
          'Incorporate Neem cake @ 150 kg/acre into the field.'
        ],
        chemical: [
          'Spray Streptocycline @ 1g + Copper Oxychloride (COC 50% WP) @ 30g in 15 liters of water.',
          'Alternatively, apply Plantomycin @ 1g/liter mixed with Copper hydroxide 53.8% DF @ 2g/liter.',
          'Drain excess standing water from paddy fields for 3-4 days to arrest bacterial proliferation.'
        ]
      },
      prevention: [
        'Use resistant varieties (e.g., Improved Samba Mahsuri, IR64, Swarna Sub1).',
        'Avoid excess urea/nitrogen fertilizer; apply nitrogen in 3-4 split doses with adequate potash (K).',
        'Disinfect seeds with Streptocycline solution (0.1g/10L) before sowing.',
        'Keep field bunds weed-free as grasses host the bacterium.'
      ],
      expertWarning: 'Do not spray during intense midday sun. Drain standing water before applying treatment. Consult your local Krishi Vigyan Kendra (KVK) officer for regional advisory.',
      sampleColor: '#d97706',
      badgeClass: 'badge-high'
    },
    {
      id: 'rice_brown_spot',
      name: 'Brown Spot',
      localName: 'भूरा धब्बा रोग / ಕಂದು ಚುಕ್ಕೆ ರೋಗ',
      crop: 'rice',
      cropDisplayName: 'Rice (धान)',
      scientificName: 'Bipolaris oryzae (Cochliobolus miyabeanus)',
      severity: 'Moderate',
      season: 'Kharif & Boro',
      priority: 'Moderate Priority',
      summary: 'A fungal disease characterized by oval or round brown spots on leaves with distinct grey or whitish centres, heavily reducing photosynthesis and grain yield.',
      symptoms: [
        'Small, circular to oval sesame-seed-sized dark brown spots on leaves and glumes.',
        'Mature spots have light brown or greyish centres with dark reddish-brown margins.',
        'Numerous spots coalesce, causing leaves to turn yellow and dry prematurely.',
        'Dark spots on grain hulls lead to discoloured, chaffy or unfilled grains.'
      ],
      visibleSigns: 'Oval, sesame-seed-shaped brown spots with distinct greyish centres surrounded by yellow halo.',
      causes: 'Fungal spores spread by wind and rain splash. Most severe in nutrient-deficient, poorly drained, sandy or acidic soils lacking potassium and silicon.',
      treatment: {
        organic: [
          'Foliar spray with Trichoderma harzianum or T. viride @ 5g/liter water.',
          'Apply 5% Neem Seed Kernel Extract (NSKE) at the first sighting of leaf lesions.',
          'Soil application of well-decomposed Farmyard Manure (FYM) enriched with beneficial microbes.'
        ],
        chemical: [
          'Spray Mancozeb 75% WP @ 2.5g/liter or Zineb 75% WP @ 2g/liter of water.',
          'In advanced stages, spray Propiconazole 25% EC (Tilt) @ 1 ml/liter or Tricyclazole 75% WP @ 0.6g/liter.',
          'Apply balanced Potash (MOP) to strengthen cell walls against fungal penetration.'
        ]
      },
      prevention: [
        'Treat seeds before sowing with Thiram or Carbendazim @ 2g/kg seed.',
        'Correct soil fertility: apply balanced NPK along with zinc sulfate (10 kg/acre).',
        'Avoid drought stress during tillering and panicle development stages.',
        'Destroy crop stubble and alternate weed hosts after harvest.'
      ],
      expertWarning: 'Brown spot often indicates nutrient deficiency in the soil. Get a soil test done for potassium and micronutrients.',
      sampleColor: '#92400e',
      badgeClass: 'badge-moderate'
    },
    {
      id: 'rice_leaf_blast',
      name: 'Leaf Blast',
      localName: 'पत्ती झुलसा / ब्लास्ट रोग / ಎಲೆ ಬ್ಲಾಸ್ಟ್ ರೋಗ',
      crop: 'rice',
      cropDisplayName: 'Rice (धान)',
      scientificName: 'Magnaporthe oryzae (Pyricularia oryzae)',
      severity: 'Critical',
      season: 'Kharif & Rabi',
      priority: 'Urgent / Critical',
      summary: 'One of the most destructive rice diseases globally. Causes characteristic diamond or spindle-shaped lesions that can scorch and destroy an entire paddy field rapidly.',
      symptoms: [
        'Initial small bluish-green flecks that enlarge rapidly into spindle-shaped lesions.',
        'Lesions have pointed ends (spindle/eye shape) with greyish-white ash centres and brown borders.',
        'Spots coalesce causing entire leaves to blast, wither, and look burned/scorched.',
        'Can progress to node blast and neck blast causing complete breakage of the panicle ("rotten neck").'
      ],
      visibleSigns: 'Eye-shaped or spindle-shaped lesions with ash-grey center and dark reddish-brown margins.',
      causes: 'Air-borne fungal conidia. Highly favoured by cloudy days, high relative humidity (>90%), night dew, cool nights (18-24°C), and excessive nitrogen application.',
      treatment: {
        organic: [
          'Spray Pseudomonas fluorescens liquid formulation @ 10ml/liter water at 7-day intervals.',
          'Spray Fermented Butter Milk (sour curd) @ 50ml/liter with 1% soap nut extract as wetting agent.'
        ],
        chemical: [
          'Spray Tricyclazole 75% WP (Beam / Baan) @ 0.6g/liter (most effective blast fungicide).',
          'Alternatively, spray Isoprothiolane 40% EC @ 1.5 ml/liter or Kasugamycin 3% SL @ 2 ml/liter.',
          'Repeat spray at 10-14 days if cool, humid weather persists.'
        ]
      },
      prevention: [
        'Seed treatment with Tricyclazole 75% WP @ 2g/kg seed or Carbendazim 50% WP @ 2g/kg.',
        'Never apply all nitrogen fertilizer at once; use leaf color chart (LCC) to regulate urea.',
        'Plant blast-resistant certified cultivars adapted to your agro-climatic zone.',
        'Maintain uniform shallow water depth in the field; do not let the soil dry out completely.'
      ],
      expertWarning: 'Neck blast can result in 100% yield loss if not caught early. Spray at 5% panicle emergence as preventive protection if blast was observed on leaves.',
      sampleColor: '#dc2626',
      badgeClass: 'badge-critical'
    },
    {
      id: 'rice_leaf_scald',
      name: 'Leaf Scald',
      localName: 'पत्ती स्कैल्ड / ಎಲೆ ಸುಟ್ಟ ರೋಗ',
      crop: 'rice',
      cropDisplayName: 'Rice (धान)',
      scientificName: 'Microdochium oryzae (Gerlachia oryzae)',
      severity: 'Moderate',
      season: 'Kharif (Late Monsoon)',
      priority: 'Moderate Priority',
      summary: 'A fungal disease causing characteristic zonate or banded water-soaked lesions from leaf tips that give a scorched, scalded appearance to the rice canopy.',
      symptoms: [
        'Lesions develop from leaf tips or edges as oblong water-soaked blotches.',
        'Alternating light brown and dark brown concentric bands give a distinctive chevron or wavy appearance.',
        'Affected leaves look scalded by boiling water and dry out from tip downwards.',
        'Panicles emerging from infected flag leaves remain sterile or poorly filled.'
      ],
      visibleSigns: 'Concentric wavy brown and tan zones resembling water scalds starting from the tip of mature leaves.',
      causes: 'Seed-borne and stubble-borne fungus. Proliferates in high humidity, heavy dew, wet weather, and densely spaced fields with close canopy closure.',
      treatment: {
        organic: [
          'Foliar spray with Pseudomonas fluorescens @ 5g/liter.',
          'Apply 10% garlic-chili extract as a broad-spectrum antifungal spray.'
        ],
        chemical: [
          'Foliar spray of Carbendazim 12% + Mancozeb 63% WP (Saaf) @ 2g/liter of water.',
          'Alternatively spray Propiconazole 25% EC @ 1ml/liter or Benomyl 50% WP @ 1g/liter.'
        ]
      },
      prevention: [
        'Hot water seed treatment at 52-54°C for 15 minutes, or seed dressing with Carbendazim (2g/kg).',
        'Ensure wider plant spacing (20cm x 15cm) to facilitate good air circulation.',
        'Avoid excess nitrogenous fertilizers.',
        'Deep summer ploughing to bury crop residue and fungal perithecia.'
      ],
      expertWarning: 'Often confused with bacterial leaf blight. Notice the concentric wavy bands in scald compared to straight stripes in bacterial blight.',
      sampleColor: '#b45309',
      badgeClass: 'badge-moderate'
    },
    {
      id: 'rice_sheath_blight',
      name: 'Sheath Blight',
      localName: 'शीथ ब्लाइट (तने का झुलसा) / ಪೊರೆ ರೋಗ',
      crop: 'rice',
      cropDisplayName: 'Rice (धान)',
      scientificName: 'Rhizoctonia solani',
      severity: 'High',
      season: 'Kharif',
      priority: 'High Priority',
      summary: 'A soil-borne fungal disease that attacks leaf sheaths near water level, creating greenish-grey serpent-like spots that climb up to flag leaves, causing lodging.',
      symptoms: [
        'Oval or oblong greenish-grey water-soaked spots appear on leaf sheaths just above the water line.',
        'Spots enlarge and turn bleached grey or white with dark reddish-brown borders ("snake skin" pattern).',
        'Lesions spread upward to upper sheaths and leaf blades.',
        'Infected tillers lodge (fall over) and panicles fail to fill properly.'
      ],
      visibleSigns: 'Bleached grey lesions with dark reddish-brown margins on lower leaf sheaths; small white to brown fungal sclerotia resting on lesions.',
      causes: 'Fungal sclerotia surviving in soil and irrigation water. Triggers include high humidity (85-100%), temperature (28-32°C), high plant density, and heavy nitrogen application.',
      treatment: {
        organic: [
          'Soil application of Trichoderma viride enriched FYM @ 2.5 kg in 100 kg FYM per acre.',
          'Spray bio-control agent Bacillus subtilis @ 5ml/liter water onto the base of hills.'
        ],
        chemical: [
          'Spray Hexaconazole 5% SC (Contaf) @ 2ml/liter directed at the lower stem/sheath level.',
          'Spray Validamycin 3% L (Sheathmar) @ 2.5ml/liter or Azoxystrobin 18.2% + Difenoconazole 11.4% SC (Amistar Top) @ 1ml/liter.',
          'Ensure the spray nozzle is directed towards the base of the rice hills.'
        ]
      },
      prevention: [
        'Avoid dense planting; maintain 20cm row spacing for aeration.',
        'Apply split doses of nitrogen along with adequate potash.',
        'Drain irrigation water intermittently to dry out sclerotia floating on water.',
        'Eradicate weed hosts (Echinochloa, Cyperus) along irrigation channels.'
      ],
      expertWarning: 'Do not spray chemicals only over top foliage; the spray must reach the lower leaf sheaths near water level where the fungus starts.',
      sampleColor: '#b91c1c',
      badgeClass: 'badge-high'
    }
  ],
  wheat: [
    {
      id: 'wheat_yellow_rust',
      name: 'Yellow Rust (Stripe Rust)',
      localName: 'पीला रतुआ / ಪಳದಿ ರಸ್ಟ್',
      crop: 'wheat',
      cropDisplayName: 'Wheat (गेहूं)',
      scientificName: 'Puccinia striiformis f. sp. tritici',
      severity: 'Critical',
      season: 'Rabi (Winter)',
      priority: 'Urgent / Critical',
      summary: 'The most dangerous winter wheat disease in Northern India. Forms bright yellow powder pustules arranged in parallel linear stripes along leaf veins.',
      symptoms: [
        'Bright yellow to orange-yellow powdery pustules (uredinia) arranged in distinct linear stripes.',
        'Stripes run parallel along the leaf veins on the upper surface of the leaf blade.',
        'When touched with fingers, yellow powder (spores) easily sticks to skin or clothes.',
        'Leaves chlorose, dry up prematurely, and grain shrivels severely.'
      ],
      visibleSigns: 'Parallel rows of bright yellow powder pustules resembling stitch lines along the leaf blade.',
      causes: 'Air-borne fungal spores blown from Himalayan foothills. Favored by cool temperatures (10-15°C), high humidity, morning fog, and persistent winter rains.',
      treatment: {
        organic: [
          'Early morning foliar spray with fermented butter milk (chaas) @ 50ml/liter water.',
          'Foliar spray with Trichoderma harzianum @ 5g/liter of water at initial symptom appearance.'
        ],
        chemical: [
          'Immediately spray Propiconazole 25% EC (Tilt / Bumper) @ 1 ml/liter (200 ml in 200 liters water/acre).',
          'Alternatively, spray Tebuconazole 25.9% EC (Folicur) @ 1.25 ml/liter or Azoxystrobin 18.2% + Difenoconazole 11.4% SC @ 1 ml/liter.',
          'Repeat spray after 15 days if cool, cloudy weather continues.'
        ]
      },
      prevention: [
        'Sow yellow rust-resistant certified varieties (e.g., HD 3086, HD 2967, DBW 187, DBW 222, PBW 725).',
        'Avoid late sowing; complete wheat sowing by November 15-20.',
        'Regularly scout fields, especially in shaded areas and near irrigation channels.',
        'Do not over-irrigate during cold, cloudy January spells.'
      ],
      expertWarning: 'CRITICAL: Yellow rust spores travel hundreds of kilometers on wind. Report yellow rust sightings immediately to your local Block Agriculture Officer or WhatsApp Helpline.',
      sampleColor: '#eab308',
      badgeClass: 'badge-critical'
    },
    {
      id: 'wheat_brown_rust',
      name: 'Brown Rust (Leaf Rust)',
      localName: 'भूरा रतुआ / ಕಂದು ರಸ್ಟ್',
      crop: 'wheat',
      cropDisplayName: 'Wheat (गेहूं)',
      scientificName: 'Puccinia triticina (Puccinia recondita)',
      severity: 'High',
      season: 'Rabi (Late Winter)',
      priority: 'High Priority',
      summary: 'Commonly occurs across all wheat-growing belts during mid-to-late season. Forms scattered round brown or orange-brown pustules randomly across the leaf surface.',
      symptoms: [
        'Small, round to oval orange-brown to cinnamon-brown pustules on leaves.',
        'Pustules are scattered randomly over the leaf surface (unlike yellow rust, they are NOT in stripes).',
        'Rust pustules burst through the leaf epidermis, releasing brown powdery spores.',
        'Infected leaves yellow and senesce early, reducing flag leaf photosynthesis.'
      ],
      visibleSigns: 'Randomly scattered orange-brown circular powder dots across the leaf blade, rub off on fingers.',
      causes: 'Fungal spores surviving on self-sown plants and alternate hosts. Favoured by moderate temperatures (15-25°C) and moisture/dew on leaves.',
      treatment: {
        organic: [
          'Foliar spray with 5% Neem Seed Kernel Extract (NSKE) or Neem oil @ 3ml/liter with mild detergent.',
          'Foliar application of bio-agent Pseudomonas fluorescens @ 5g/liter.'
        ],
        chemical: [
          'Spray Propiconazole 25% EC @ 1ml/liter or Mancozeb 75% WP @ 2.5g/liter of water.',
          'Spray Tebuconazole 25% WG @ 1g/liter if disease reaches economic threshold (5-10% leaf area).'
        ]
      },
      prevention: [
        'Adopt resistant varieties recommended for your specific zone.',
        'Timely sowing to avoid exposure to warmer March temperatures.',
        'Avoid excessive urea; balance with phosphorus and potassium.',
        'Destroy volunteer wheat plants in summer fallow.'
      ],
      expertWarning: 'Notice that Brown Rust pustules are scattered randomly, while Yellow Rust pustules are strictly aligned in neat stripes.',
      sampleColor: '#b45309',
      badgeClass: 'badge-high'
    },
    {
      id: 'wheat_stem_rust',
      name: 'Stem Rust (Black Rust)',
      localName: 'काला रतुआ / ಕಪ್ಪು ರಸ್ಟ್',
      crop: 'wheat',
      cropDisplayName: 'Wheat (गेहूं)',
      scientificName: 'Puccinia graminis f. sp. tritici',
      severity: 'Critical',
      season: 'Rabi (Late Season / Warm Spring)',
      priority: 'Urgent / Critical',
      summary: 'Also called Black Rust. Attacks wheat stems, leaf sheaths, and spikes, creating large reddish-brown to dark black pustules that rupture the stem epidermis, causing lodging and grain destruction.',
      symptoms: [
        'Large, elongated reddish-brown pustules on stems, leaf sheaths, and glumes.',
        'Pustules rupture the outer stem tissue, leaving ragged torn epidermal edges.',
        'Late in season, pustules turn deep brownish-black as black teliospores form.',
        'Stems weaken dramatically, causing severe lodging and brittle straw.'
      ],
      visibleSigns: 'Rough, torn dark brown to black pustules rupturing along the stems and sheaths.',
      causes: 'Warm temperatures (20-30°C) with dew or rain showers late in the Rabi season. Common in Central and Peninsular India.',
      treatment: {
        organic: [
          'Foliar spray with cow urine extract (1:10 dilution with water) mixed with neem leaf extract.',
          'Application of bio-fungicide Ampelomyces quisqualis or Trichoderma @ 5g/liter.'
        ],
        chemical: [
          'Foliar spray of Propiconazole 25% EC @ 1ml/liter or Tebuconazole 25.9% EC @ 1ml/liter.',
          'Apply spray thoroughly to ensure stems and leaf sheaths are well wetted.',
          'Apply early before black teliospores harden on stems.'
        ]
      },
      prevention: [
        'Plant certified stem-rust resistant cultivars (e.g., HD 2967, WH 1105, MP 3288).',
        'Eradicate barberry bushes (Berberis spp.) near fields as they serve as alternate hosts.',
        'Timely irrigation and harvesting before peak summer heat.',
        'Avoid late sowing in warmer central and southern zones.'
      ],
      expertWarning: 'Stem rust can destroy an entire field within 2-3 weeks if temperatures rise above 20°C. Immediate chemical protection is necessary upon first detection.',
      sampleColor: '#451a03',
      badgeClass: 'badge-critical'
    },
    {
      id: 'wheat_powdery_mildew',
      name: 'Powdery Mildew',
      localName: 'चूर्णिल आसिता / ಬೂದಿ ರೋಗ',
      crop: 'wheat',
      cropDisplayName: 'Wheat (गेहूं)',
      scientificName: 'Blumeria graminis f. sp. tritici',
      severity: 'Moderate',
      season: 'Rabi',
      priority: 'Moderate Priority',
      summary: 'A fungal disease producing white, fluffy cotton-like powdery patches on upper leaf surfaces, sheaths, and stems, severely reducing photosynthetic efficiency.',
      symptoms: [
        'White to greyish-white powdery, talc-like patches on leaves and lower stems.',
        'Patches enlarge, coalesce, and turn dull grey or yellowish-brown.',
        'Tiny black specks (cleistothecia fruiting bodies) appear embedded in the powdery mat later in season.',
        'Leaves turn yellow, wither, and die back from the tip.'
      ],
      visibleSigns: 'White, flour-like powdery growth on upper leaf surfaces and lower canopy stems.',
      causes: 'Favoured by dense lush crop growth, shaded environments, high relative humidity (85-100%), and moderate temperatures (15-22°C). Unlike rusts, free water inhibits germination; high humidity is sufficient.',
      treatment: {
        organic: [
          'Foliar spray with Baking Soda (Sodium Bicarbonate) @ 3g/liter + 2ml liquid soap as spreader.',
          'Foliar spray with Milk solution (1 part milk : 9 parts water) in bright morning sunlight.',
          'Wettable Sulfur (Sulfex 80% WP) @ 2.5g/liter.'
        ],
        chemical: [
          'Spray Propiconazole 25% EC @ 1ml/liter or Triadimefon 25% WP (Bayleton) @ 1g/liter.',
          'Alternatively spray Hexaconazole 5% EC @ 1.5ml/liter of water.',
          'Spray early in the morning when wind velocity is low.'
        ]
      },
      prevention: [
        'Avoid overly dense sowing; maintain proper seed rate (40 kg/acre for timely sown wheat).',
        'Avoid excessive nitrogen fertilization which creates thick, succulent canopies.',
        'Ensure good sunlight penetration by orienting rows north-to-south.',
        'Select mildew-tolerant wheat cultivars.'
      ],
      expertWarning: 'Avoid sulfur sprays if temperature exceeds 30°C to prevent leaf scorching. Apply in morning hours.',
      sampleColor: '#78716c',
      badgeClass: 'badge-moderate'
    }
  ]
};

/**
 * Helper to get all diseases as a flat array
 */
export function getAllDiseases() {
  return [...DISEASES_DATABASE.rice, ...DISEASES_DATABASE.wheat];
}

/**
 * Helper to look up disease by ID
 */
export function getDiseaseById(id) {
  const all = getAllDiseases();
  return all.find(d => d.id === id) || null;
}

/**
 * Helper to get diseases by crop
 */
export function getDiseasesByCrop(crop) {
  return DISEASES_DATABASE[crop.toLowerCase()] || [];
}
