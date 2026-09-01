export type AnalysisRecord = {
  id: string;
  question: string;
  inputMode: 'Single scene' | 'Bi-temporal' | 'Optical + SAR';
  workflow: string;
  status: 'Completed' | 'Processing' | 'Abstained';
  createdAt: string;
};

export const seedHistory: AnalysisRecord[] = [
  {
    id: 'SQ-260901-001',
    question: 'Where has the built-up area changed between these dates?',
    inputMode: 'Bi-temporal',
    workflow: 'Change analysis',
    status: 'Completed',
    createdAt: '01 Sep 2026, 11:42',
  },
  {
    id: 'SQ-260831-014',
    question: 'Map the visible water and vegetation classes.',
    inputMode: 'Optical + SAR',
    workflow: 'Sensor fusion',
    status: 'Completed',
    createdAt: '31 Aug 2026, 18:09',
  },
  {
    id: 'SQ-260831-008',
    question: 'Describe the settlement pattern in this scene.',
    inputMode: 'Single scene',
    workflow: 'Scene understanding',
    status: 'Completed',
    createdAt: '31 Aug 2026, 15:21',
  },
];

export const resultData = {
  answer:
    'Two concentrated expansion regions are visible. The larger region is north-east of the scene centre, while a smaller region appears to the south-west. Together they account for approximately 12.48 km² of detected change.',
  changedArea: '12.48 km²',
  changedPercent: '8.7%',
  regions: '02',
  confidence: 91,
};
