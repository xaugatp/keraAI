/**
 * Friendly names for the backend contract.
 *
 * WHY: `schema.d.ts` is generated from `Backend/openapi.json` (`npm run gen:api`) and its names
 * are awkward (`components["schemas"]["ModelInfoOut"]`). Everything in the UI imports from here,
 * never from `schema.d.ts` directly, so a backend rename touches one file. Every type below is
 * DERIVED from the generated schema (nothing is re-typed by hand), so `gen:api` + `tsc` surfaces
 * contract drift as a compile error.
 */
import type { components, operations } from './schema';

type Schemas = components['schemas'];

// --- Core resources --------------------------------------------------------------------------

/** Full analysis: `POST .../predict` (201) and `GET /analyses/{id}`. */
export type AnalysisDetail = Schemas['AnalysisDetail'];
/** One row of the history list. */
export type AnalysisSummary = Schemas['AnalysisSummary'];
/** Generic page envelope `{items, page, page_size, total, total_pages}`. */
export type Page<T> = Omit<Schemas['Page_AnalysisSummary_'], 'items'> & { items: T[] };
/** What `GET /analyses` returns. */
export type AnalysisSummaryPage = Page<AnalysisSummary>;
/** One entry of `GET /models` (includes `is_placeholder`). */
export type ModelInfo = Schemas['ModelInfoOut'];

// --- Pieces of an AnalysisDetail -------------------------------------------------------------

export type TreeDetails = Schemas['TreeDetails'];
export type LeafSegDetails = Schemas['LeafSegDetails'];
export type LeafDiseaseDetails = Schemas['LeafDiseaseDetails'];
export type DiseaseChannelMetrics = Schemas['DiseaseChannelMetrics'];
/** `AnalysisDetail.details` without the `null` (a failed row has `details === null`). */
export type AnalysisDetails = NonNullable<AnalysisDetail['details']>;
export type Prediction = Schemas['Prediction'];
export type ImageLinks = Schemas['ImageLinks'];
export type TimingsMs = Schemas['TimingsMs'];
/** Every field is optional/nullable: the client may have sent any subset. The whole block is null when none was sent. */
export type GeoLocation = Schemas['GeoLocation'];
export type TreeProbability = Schemas['TreeProbability'];
export type LeafSegThresholds = Schemas['LeafSegThresholds'];
export type LeafDiseaseThresholds = Schemas['LeafDiseaseThresholds'];
/** RFC 9457 error body (`application/problem+json`). */
export type ProblemDetail = Schemas['ProblemDetail'];

// --- Enumerations ----------------------------------------------------------------------------

export type ModelKey = AnalysisDetail['model_key']; // 'tree_classification' | 'leaf_segmentation' | 'leaf_disease'
/** All three models have a `/predict` route. */
export type PredictModelKey = ModelKey;
/** Where a stored analysis came from. `'sample'` rows are the public pre-computed samples. */
export type AnalysisSource = AnalysisDetail['source']; // 'upload' | 'camera' | 'sample'
/** What the client may send as `source` on predict (a client can never create a 'sample'). */
export type UploadSource = Exclude<AnalysisSource, 'sample'>;
export type AnalysisStatus = AnalysisDetail['status']; // 'completed' | 'failed'
export type TreeVerdict = TreeDetails['verdict']; // 'banana_tree' | 'not_banana_tree' | 'uncertain'
export type ModelStatus = ModelInfo['status']; // 'ready' | 'unavailable'

type ListQuery = NonNullable<operations['list_analyses_api_v1_analyses_get']['parameters']['query']>;
type ImageQuery = NonNullable<
  operations['get_analysis_image_api_v1_analyses__analysis_id__image_get']['parameters']['query']
>;
/** History filter: 'mine' = this device, 'samples' = public samples, 'all_visible' = both. */
export type Scope = NonNullable<ListQuery['scope']>;
/** Which stored image the `/image` route serves. */
export type Variant = NonNullable<ImageQuery['variant']>;

/**
 * Leaf-segmentation verdicts. NOT in the OpenAPI schema (`Prediction.label` is a plain
 * `string | null`); this is the Model 2 contract from FRONTEND_INTEGRATION_SPEC §A. Treat any
 * other string defensively.
 */
export type LeafSegLabel = 'affected' | 'healthy' | 'no_leaf';

/**
 * Leaf-disease diagnoses. NOT in the OpenAPI schema (`Prediction.label` is a plain
 * `string | null`); this is the Model 3 contract from ADR 0019. Treat any other string
 * defensively.
 */
export type LeafDiseaseLabel = 'black_sigatoka' | 'yellow_sigatoka' | 'healthy' | 'no_leaf';

// --- Type guards -----------------------------------------------------------------------------

/**
 * Narrow `analysis.details` to the tree shape. Accepts `null`/`undefined` because a failed
 * analysis has `details === null` — the guard simply returns false for it.
 */
export function isTreeDetails(details: AnalysisDetail['details'] | undefined): details is TreeDetails {
  return details != null && details.kind === 'tree_classification';
}

/** Narrow `analysis.details` to the leaf-segmentation shape (false for `null`/`undefined`). */
export function isLeafSegDetails(
  details: AnalysisDetail['details'] | undefined,
): details is LeafSegDetails {
  return details != null && details.kind === 'leaf_segmentation';
}

/** Narrow `analysis.details` to the leaf-disease shape (false for `null`/`undefined`). */
export function isLeafDiseaseDetails(
  details: AnalysisDetail['details'] | undefined,
): details is LeafDiseaseDetails {
  return details != null && details.kind === 'leaf_disease';
}
