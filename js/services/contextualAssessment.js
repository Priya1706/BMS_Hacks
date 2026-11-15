/**
 * Crop-specific, user-reported observations for the optional second-stage
 * assessment. This module never changes model scores or disease predictions.
 */

const QUESTIONS = {
  rice: {
    symptoms: [
      ['edge_stripes', 'Yellow or white wavy stripes along leaf edges'],
      ['brown_spots', 'Small round or oval brown spots on leaves'],
      ['gray_centers', 'Brown spots with pale or grey centers'],
      ['water_soaked', 'Water-soaked areas or milky droplets on leaves'],
      ['scorched_tips', 'Scorched-looking leaf tips or chevron-like bands'],
      ['sheath_lesions', 'Oval grey-brown lesions on lower leaf sheaths'],
      ['no_visible_symptoms', 'No unusual symptoms noticed']
    ],
    irrigation: [
      ['standing', 'Standing water most of the time'],
      ['intermittent', 'Intermittent wetting and drying'],
      ['limited', 'Little or no irrigation recently'],
      ['unsure', 'Not sure']
    ]
  },
  wheat: {
    symptoms: [
      ['scattered_brown_pustules', 'Scattered orange or brown powdery dots on leaves'],
      ['yellow_stripes', 'Yellow powdery dots arranged in rows or stripes'],
      ['dark_stem_pustules', 'Large dark or black lesions on stems or sheaths'],
      ['white_powder', 'White flour-like patches on leaves or stems'],
      ['yellow_drying', 'Leaves turning yellow or drying early'],
      ['no_visible_symptoms', 'No unusual symptoms noticed']
    ],
    irrigation: [
      ['recent', 'Irrigated recently'],
      ['not_recent', 'Not irrigated recently'],
      ['unsure', 'Not sure']
    ]
  }
};

const DISEASE_CUES = {
  rice_bacterial_leaf_blight: ['edge_stripes', 'water_soaked'],
  rice_brown_spot: ['brown_spots', 'gray_centers'],
  rice_healthy: ['no_visible_symptoms'],
  rice_leaf_scald: ['scorched_tips'],
  rice_sheath_blight: ['sheath_lesions'],
  wheat_stem_rust: ['dark_stem_pustules'],
  wheat_brown_rust: ['scattered_brown_pustules'],
  wheat_yellow_rust: ['yellow_stripes'],
  wheat_powdery_mildew: ['white_powder']
};

const DISTINCTIVE_CONFLICTS = {
  rice_bacterial_leaf_blight: ['brown_spots', 'sheath_lesions'],
  rice_brown_spot: ['edge_stripes', 'water_soaked', 'sheath_lesions'],
  rice_healthy: ['edge_stripes', 'brown_spots', 'water_soaked', 'scorched_tips', 'sheath_lesions'],
  rice_leaf_scald: ['water_soaked', 'sheath_lesions'],
  rice_sheath_blight: ['edge_stripes', 'brown_spots', 'scorched_tips'],
  wheat_stem_rust: ['yellow_stripes', 'white_powder'],
  wheat_brown_rust: ['yellow_stripes', 'dark_stem_pustules', 'white_powder'],
  wheat_yellow_rust: ['scattered_brown_pustules', 'dark_stem_pustules', 'white_powder'],
  wheat_powdery_mildew: ['scattered_brown_pustules', 'yellow_stripes', 'dark_stem_pustules']
};

export function getQuestionnaireOptions(crop) {
  return QUESTIONS[String(crop || '').toLowerCase()] || null;
}

export function assessFieldObservations(disease, answers) {
  const selected = Array.isArray(answers?.symptoms) ? answers.symptoms : [];
  const cues = DISEASE_CUES[disease?.id] || [];
  const conflicts = DISTINCTIVE_CONFLICTS[disease?.id] || [];
  const diseaseRelevant = selected.filter(item =>
    cues.includes(item) || conflicts.includes(item) || item === 'no_visible_symptoms'
  );
  const matching = diseaseRelevant.filter(item => cues.includes(item));
  const conflicting = diseaseRelevant.filter(item => conflicts.includes(item));
  const meaningfulContext = ['affectedArea', 'growthStage', 'duration', 'spread', 'weather', 'irrigation']
    .filter(key => answers?.[key] && answers[key] !== 'unsure');

  if (!disease || (diseaseRelevant.length === 0 && meaningfulContext.length < 3)) {
    return {
      status: 'insufficient',
      title: 'Insufficient information for a meaningful comparison',
      explanation: 'Select the visible symptoms you noticed and, if possible, answer a few field questions. The AI prediction and its confidence remain unchanged.'
    };
  }

  if (selected.includes('no_visible_symptoms') && disease.id !== 'rice_healthy') {
    conflicting.push('no_visible_symptoms');
  }
  if (conflicting.length > 0 || (diseaseRelevant.length > 0 && matching.length === 0 && !diseaseRelevant.includes('no_visible_symptoms'))) {
    const detail = conflicting.length
      ? 'Some selected signs differ from the characteristic signs listed for this prediction.'
      : 'The selected signs do not specifically match the characteristic signs listed for this prediction.';
    return {
      status: 'misaligned',
      title: 'Some observations do not align with the AI prediction',
      explanation: `${detail} This comparison is informational and does not confirm or rule out a disease.`
    };
  }

  if (matching.length > 0) {
    return {
      status: 'consistent',
      title: 'Observations are consistent with the AI prediction',
      explanation: 'At least one symptom you selected is among the characteristic signs associated with this prediction. Agreement does not confirm a diagnosis; the original model confidence is unchanged.'
    };
  }

  return {
    status: 'insufficient',
    title: 'Insufficient information for a meaningful comparison',
    explanation: 'The field details provide context, but there are not enough disease-specific symptoms to compare. The AI prediction and its confidence remain unchanged.'
  };
}
