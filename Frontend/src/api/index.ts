/**
 * Public surface of the API layer. Import from `../api` (not from the individual files) so
 * internals can move without touching views.
 */
export { API_BASE_URL, API_PREFIX } from './config.ts';
export { getClientId } from './clientId.ts';
export { ACCEPTED_IMAGE_TYPES, MAX_UPLOAD_BYTES, MAX_UPLOAD_MB } from './limits.ts';
export {
  AbortedError,
  ApiError,
  NetworkError,
  isOfflineError,
  requestIdFor,
  userMessageFor,
} from './errors.ts';
export type { FieldError } from './errors.ts';
export { predict } from './upload.ts';
export type { PredictInput, PredictOptions, UploadProgress } from './upload.ts';
export { deleteAnalysis, getAnalysis, listAnalyses } from './analyses.ts';
export type { ListAnalysesParams } from './analyses.ts';
export { getModels } from './models.ts';
export { getReady } from './health.ts';
export type { ReadyComponent, ReadyResult } from './health.ts';
export { absoluteImageUrl } from './urls.ts';
export { validateImageFile } from './validateImage.ts';
export type { PickedFileLike } from './validateImage.ts';
export { resolveCapturedAt } from './capturedAt.ts';
export type { CapturedAtInputs } from './capturedAt.ts';
export { isLeafSegDetails, isTreeDetails } from './types.ts';
export type {
  AnalysisDetail,
  AnalysisDetails,
  AnalysisSource,
  AnalysisStatus,
  AnalysisSummary,
  AnalysisSummaryPage,
  GeoLocation,
  ImageLinks,
  LeafSegDetails,
  LeafSegLabel,
  LeafSegThresholds,
  ModelInfo,
  ModelKey,
  ModelStatus,
  Page,
  PredictModelKey,
  Prediction,
  ProblemDetail,
  Scope,
  TimingsMs,
  TreeDetails,
  TreeProbability,
  TreeVerdict,
  UploadSource,
  Variant,
} from './types.ts';
