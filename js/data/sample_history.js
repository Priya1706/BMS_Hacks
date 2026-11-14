/**
 * Sample historical diagnosis records for initial offline presentation
 */

export const INITIAL_SAMPLE_HISTORY = [
  {
    id: 'diag_1001',
    timestamp: '2026-11-14T09:30:00Z',
    displayDate: '14 Nov 2026, 09:30 AM',
    crop: 'rice',
    cropDisplayName: 'Rice (धान)',
    diseaseId: 'rice_bacterial_leaf_blight',
    diseaseName: 'Bacterial Leaf Blight',
    confidence: 94.8,
    severity: 'High',
    status: 'Confirmed Infection',
    farmerName: 'Ramesh Patel',
    region: 'Karnal, Haryana',
    notes: 'Observed yellow stripes drying along leaf margins after heavy monsoon showers.',
    imageThumbnail: 'assets/images/sample_leaf_rice.svg'
  },
  {
    id: 'diag_1002',
    timestamp: '2026-11-13T15:45:00Z',
    displayDate: '13 Nov 2026, 03:45 PM',
    crop: 'wheat',
    cropDisplayName: 'Wheat (गेहूं)',
    diseaseId: 'wheat_yellow_rust',
    diseaseName: 'Yellow Rust (Stripe Rust)',
    confidence: 97.2,
    severity: 'Critical',
    status: 'Urgent Action Required',
    farmerName: 'Gurpreet Singh',
    region: 'Ludhiana, Punjab',
    notes: 'Parallel yellow stripes noticed on upper leaves during early morning fog check.',
    imageThumbnail: 'assets/images/sample_leaf_wheat.svg'
  },
  {
    id: 'diag_1003',
    timestamp: '2026-11-11T11:15:00Z',
    displayDate: '11 Nov 2026, 11:15 AM',
    crop: 'rice',
    cropDisplayName: 'Rice (धान)',
    diseaseId: 'rice_leaf_blast',
    diseaseName: 'Leaf Blast',
    confidence: 91.5,
    severity: 'Critical',
    status: 'Confirmed Infection',
    farmerName: 'Suresh Gowda',
    region: 'Mandya, Karnataka',
    notes: 'Spindle-shaped ash-centered eye lesions spreading fast on tiller leaves.',
    imageThumbnail: 'assets/images/sample_leaf_rice.svg'
  },
  {
    id: 'diag_1004',
    timestamp: '2026-11-08T16:20:00Z',
    displayDate: '08 Nov 2026, 04:20 PM',
    crop: 'wheat',
    cropDisplayName: 'Wheat (गेहूं)',
    diseaseId: 'wheat_powdery_mildew',
    diseaseName: 'Powdery Mildew',
    confidence: 89.4,
    severity: 'Moderate',
    status: 'Early Stage Detection',
    farmerName: 'Anil Deshmukh',
    region: 'Nashik, Maharashtra',
    notes: 'White powdery patches on shaded lower stems following humid mornings.',
    imageThumbnail: 'assets/images/sample_leaf_wheat.svg'
  }
];
