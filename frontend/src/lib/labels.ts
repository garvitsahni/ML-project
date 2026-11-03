export interface FeatureLabel {
  label: string
  help: string
}

const LABELS: Record<string, FeatureLabel> = {
  person_count: {
    label: 'People',
    help: 'How many people the detector found in this view.',
  },
  density_per_mpx: {
    label: 'Crowdedness',
    help: 'People per megapixel — how packed the picture is.',
  },
  grid_concentration: {
    label: 'Bunched together',
    help: 'Share of people standing in the busiest part of the grid.',
  },
  spread_x: {
    label: 'Spread left–right',
    help: 'How widely people are spread across the width.',
  },
  spread_y: {
    label: 'Spread up–down',
    help: 'How widely people are spread across the height.',
  },
  grid_entropy: {
    label: 'Evenness',
    help: 'How evenly people are spread across the grid.',
  },
  mean_speed_pxf: {
    label: 'Speed',
    help: 'Average movement in pixels per frame.',
  },
  speed_std_pxf: {
    label: 'Speed varies',
    help: 'How much people differ in speed.',
  },
  directional_consistency: {
    label: 'Moving together',
    help: 'How much people move in the same direction.',
  },
  direction_entropy: {
    label: 'Direction spread',
    help: 'How scattered the walking directions are.',
  },
  flow_mean_magnitude: {
    label: 'Motion',
    help: 'Average amount of movement in the picture.',
  },
  flow_std_pxf: {
    label: 'Motion varies',
    help: 'How uneven the movement is across the picture.',
  },
  track_count: {
    label: 'People tracked',
    help: 'How many people were followed across the window.',
  },
  track_persistence: {
    label: 'Staying in view',
    help: 'How long people stay visible across the window.',
  },
  density_delta: {
    label: 'People changing',
    help: 'How much the crowd size changed over the window.',
  },
}

export function featureLabel(name: string): FeatureLabel {
  return (
    LABELS[name] ?? {
      label: name.replace(/_/g, ' '),
      help: '',
    }
  )
}
