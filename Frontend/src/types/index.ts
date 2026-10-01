export interface BoundingBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface SoftmaxScore {
  className: string;
  scientificName?: string;
  probability: number;
  color: string;
}

export interface Stage1Detection {
  batchId: string;
  timestamp: string;
  status: 'valid' | 'rejected';
  targetFound: boolean;
  species: string;
  family: string;
  confidence: number;
  boundingBox: BoundingBox;
  resolution: string;
  colorSpace: string;
  foliageDensity: string;
  pseudostemIntegrity: number;
  luminanceBalance: number;
  distribution: SoftmaxScore[];
  architecture: {
    modelName: string;
    topology: string;
    latencyMs: number;
    hardware: string;
    endpoint: string;
  };
  rejectionReason?: string;
  rejectionDistribution?: SoftmaxScore[];
}

export interface LesionCluster {
  id: string;
  title: string;
  stage: string;
  confidence: number;
  areaPercent: number;
  xPercent: number;
  yPercent: number;
  type: 'necrotic' | 'chlorotic' | 'marginal';
  notes: string;
}

export interface Stage2Diagnosis {
  sampleId: string;
  blockLocation: string;
  scanTime: string;
  severity: 'severe' | 'moderate' | 'low' | 'healthy';
  classificationClass: string;
  pathogenCommon: string;
  pathogenScientific: string;
  certainty: number;
  psiSeverityIndex: number;
  infectedAreaPercent: number;
  spectralRatios: {
    healthyTissue: number;
    chlorosisHalo: number;
    necroticLesions: number;
    totalInfected: number;
  };
  model3: {
    modelName: string;
    role: string;
    endpoint: string;
    disease: string;
    confidence: number;
    latencyMs: number;
    status: string;
    requestPayload?: string;
    responseJson?: string;
  };
  model4: {
    modelName: string;
    role: string;
    endpoint: string;
    infectedAreaPercent: number;
    diceScore: number;
    latencyMs: number;
    status: string;
    requestPayload?: string;
    responseJson?: string;
  };
  recommendations: {
    chemical: string;
    cultural: string;
    epidemicRisk: string;
    reInspectionHours: string;
  };
  controlSample: {
    title: string;
    confidence: number;
    latencySec: number;
    imageUrl: string;
  };
  telemetry: {
    architecture: string;
    inputTensor: string;
    latencyMs: number;
    diceScore: number;
    endpoint: string;
  };
  lesions: LesionCluster[];
  imageUrl: string;
}

export type ViewTab =
  | 'home'
  | 'detect'
  | 'detect-disease'
  | 'plant-result'
  | 'leaf-detect'
  | 'leaf-result'
  | 'history'
  | 'about'
  // aliases for backwards compatibility
  | 'detection-workspace'
  | 'stage1-result'
  | 'leaf-analysis'
  | 'analysis-history'
  | 'about-models';
