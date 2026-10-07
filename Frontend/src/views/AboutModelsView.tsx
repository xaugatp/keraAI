import React, { useState } from 'react';
import { ViewTab } from '../types';
import { ASSETS, benchmarkDatasets } from '../data/mockData';
import { FutureScopeSection } from '../components/FutureScopeSection';

interface AboutModelsViewProps {
  onNavigate: (tab: ViewTab) => void;
}

export const AboutModelsView: React.FC<AboutModelsViewProps> = ({ onNavigate }) => {
  const [activeModelTab, setActiveModelTab] = useState<'model1' | 'model2' | 'model3' | 'pipeline'>('model1');
  const [copiedApiCmd, setCopiedApiCmd] = useState<string | null>(null);

  // Honest, plain-text summary of the real training data (no fabricated DOI/citation —
  // these are private Roboflow exports, not a published dataset).
  const datasetSummary = benchmarkDatasets
    .map((ds) => `${ds.name}: ${ds.samples}, classes: ${ds.classes} (${ds.source})`)
    .join('\n');

  const copyText = (text: string, key: string) => {
    if (navigator.clipboard) {
      navigator.clipboard.writeText(text);
      setCopiedApiCmd(key);
      setTimeout(() => setCopiedApiCmd(null), 2000);
    }
  };

  return (
    <div className="w-full max-w-7xl mx-auto px-6 lg:px-12 py-10 flex flex-col gap-10">
      {/* Title & Research Initiative Header */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2 text-xs font-mono uppercase text-[#3d4a42] font-semibold">
          <span>RESEARCH INITIATIVE</span>
          <span>/</span>
          <span className="text-[#006948] font-bold">COMPUTER VISION BENCHMARKS</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-[#131b2e] tracking-tight">
          About KERA AI Models &amp; Architecture
        </h1>
        <p className="text-base text-[#3d4a42] max-w-3xl leading-relaxed">
          Real training data, real figures and real checkpoints behind Model 1 (tree classification), Model 2 (leaf
          segmentation) and Model 3 (leaf disease identification) — the same three models the FastAPI backend serves.
        </p>
      </div>

      {/* Model Selection Tabs */}
      <div className="flex items-center gap-2 border-b border-[#dae2fd] pb-1 overflow-x-auto">
        <button
          type="button"
          onClick={() => setActiveModelTab('model1')}
          className={`px-4 py-2.5 rounded-t-xl text-xs font-bold transition-all cursor-pointer whitespace-nowrap ${
            activeModelTab === 'model1'
              ? 'bg-white text-[#006948] border-t-2 border-[#006948] shadow-sm'
              : 'text-[#3d4a42] hover:text-[#131b2e]'
          }`}
        >
          Model 1: Tree Classifier
        </button>

        <button
          type="button"
          onClick={() => setActiveModelTab('model2')}
          className={`px-4 py-2.5 rounded-t-xl text-xs font-bold transition-all cursor-pointer whitespace-nowrap ${
            activeModelTab === 'model2'
              ? 'bg-white text-[#006948] border-t-2 border-[#006948] shadow-sm'
              : 'text-[#3d4a42] hover:text-[#131b2e]'
          }`}
        >
          Model 2: Leaf Segmentation
        </button>

        <button
          type="button"
          onClick={() => setActiveModelTab('model3')}
          className={`px-4 py-2.5 rounded-t-xl text-xs font-bold transition-all cursor-pointer whitespace-nowrap ${
            activeModelTab === 'model3'
              ? 'bg-white text-[#006948] border-t-2 border-[#006948] shadow-sm'
              : 'text-[#3d4a42] hover:text-[#131b2e]'
          }`}
        >
          Model 3: Leaf Disease ID
        </button>

        <button
          type="button"
          onClick={() => setActiveModelTab('pipeline')}
          className={`px-4 py-2.5 rounded-t-xl text-xs font-bold transition-all cursor-pointer whitespace-nowrap ${
            activeModelTab === 'pipeline'
              ? 'bg-white text-[#006948] border-t-2 border-[#006948] shadow-sm'
              : 'text-[#3d4a42] hover:text-[#131b2e]'
          }`}
        >
          How It Runs
        </button>
      </div>

      {/* MODEL 1 CONTENT */}
      {activeModelTab === 'model1' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          <div className="lg:col-span-7 flex flex-col gap-6">
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col gap-4">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <h3 className="text-lg font-bold text-[#131b2e]">
                  Banana Tree Classifier (YOLOv8n-cls)
                </h3>
                <span className="px-2.5 py-0.5 rounded-full bg-[#85f8c4] text-[#002114] font-mono text-xs font-bold">
                  94.3% top-1 accuracy
                </span>
              </div>
              <p className="text-xs text-[#3d4a42] leading-relaxed">
                A binary image classifier that answers one question from a single photo: is this a
                banana tree, or not? It is deliberately a <em>classifier</em>, not an object
                detector — the reference dataset's bounding boxes always span the full frame (no
                real localization signal), so a detector would add complexity without adding
                accuracy. See <em>ADR 0013</em> for the full reasoning. It gives a confidence score
                and explicitly reports &ldquo;uncertain&rdquo; below a 0.60 threshold rather than
                forcing a guess.
              </p>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-[11px] text-center pt-1">
                <div className="p-2.5 rounded bg-[#f2f3ff] border border-[#dae2fd]">
                  <div className="font-bold text-[#131b2e]">708</div>
                  <div className="text-[10px] text-[#3d4a42]">Source images (Roboflow)</div>
                </div>
                <div className="p-2.5 rounded bg-[#f2f3ff] border border-[#dae2fd]">
                  <div className="font-bold text-[#131b2e]">496 / 106 / 106</div>
                  <div className="text-[10px] text-[#3d4a42]">Train / Val / Test</div>
                </div>
                <div className="p-2.5 rounded bg-[#f2f3ff] border border-[#dae2fd]">
                  <div className="font-bold text-[#131b2e]">1.44M</div>
                  <div className="text-[10px] text-[#3d4a42]">Trainable parameters</div>
                </div>
                <div className="p-2.5 rounded bg-[#f2f3ff] border border-[#dae2fd]">
                  <div className="font-bold text-[#131b2e]">CPU-only</div>
                  <div className="text-[10px] text-[#3d4a42]">Intel Core Ultra 7, no CUDA</div>
                </div>
              </div>

              {/* Data integrity: the leakage finding + fix, real figures */}
              <div className="flex flex-col gap-2 pt-2">
                <span className="font-mono text-xs font-bold text-[#131b2e] uppercase">
                  Data integrity: fixing a real train/test leak before training
                </span>
                <p className="text-[11px] text-[#3d4a42] leading-relaxed">
                  Roboflow's own pre-made split put different crops of the <em>same</em> capture
                  session in both train and test (88 sessions shared between train/valid, 53 between
                  train/test) — the model could have memorized backgrounds instead of learning
                  &ldquo;banana tree&rdquo;. Fixed by re-splitting at the capture-session level
                  (grouped split, seed-searched for best class balance), then oversampling the
                  minority Non-Banana class (82 &rarr; 207 in training) so the classifier isn't
                  just biased toward predicting the majority class.
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <img
                    src={ASSETS.model1ClassBalance}
                    alt="Class counts per split after the group-aware re-split, by class (Banana vs Non-Banana)"
                    className="w-full rounded-xl border border-[#dae2fd] bg-white"
                  />
                  <img
                    src={ASSETS.model1Oversampling}
                    alt="Training-set class counts before vs after oversampling the Non-Banana minority class"
                    className="w-full rounded-xl border border-[#dae2fd] bg-white"
                  />
                </div>
              </div>

              {/* Optimizer sweep, real figure */}
              <div className="flex flex-col gap-2 pt-2">
                <span className="font-mono text-xs font-bold text-[#131b2e] uppercase">
                  Optimizer selection: a short bake-off, not an assumption
                </span>
                <p className="text-[11px] text-[#3d4a42] leading-relaxed">
                  AdamW and SGD were each given an identical short training budget and compared on
                  validation accuracy before committing to a full run. SGD-momentum won (0.943 vs
                  0.934) and was used for the full training run below.
                </p>
                <img
                  src={ASSETS.model1OptimizerSweep}
                  alt="Validation accuracy of AdamW vs SGD after an identical short training budget"
                  className="w-full max-w-sm mx-auto rounded-xl border border-[#dae2fd] bg-white"
                />
              </div>

              {/* Real training curves */}
              <div className="flex flex-col gap-2 pt-2">
                <span className="font-mono text-xs font-bold text-[#131b2e] uppercase">
                  Training &amp; validation curves (real run)
                </span>
                <img
                  src={ASSETS.model1TrainingCurves}
                  alt="Real training/validation loss and accuracy curves from the YOLOv8n-cls training run"
                  className="w-full rounded-xl border border-[#dae2fd] bg-white"
                />
                <p className="text-[11px] text-[#3d4a42] leading-relaxed">
                  Early stopping (patience=15) found its best checkpoint at epoch 2 of a 60-epoch
                  budget — the pretrained backbone adapts fast on a dataset this size, and further
                  epochs did not beat it on validation accuracy.
                </p>
              </div>

              {/* Real confusion matrix + classification report */}
              <div className="flex flex-col gap-2 pt-2">
                <span className="font-mono text-xs font-bold text-[#131b2e] uppercase">
                  Test-set results (N=106, held out, never seen during training)
                </span>
                <img
                  src={ASSETS.model1ConfusionMatrix}
                  alt="Confusion matrix on the held-out test set"
                  className="w-full max-w-sm mx-auto rounded-xl border border-[#dae2fd] bg-white"
                />
                <div className="grid grid-cols-2 gap-2 font-mono text-[11px] text-center pt-1">
                  <div className="p-2.5 rounded bg-[#85f8c4]/30 border border-[#85f8c4]">
                    <div className="font-bold text-[#006948]">97% / 97% / 97%</div>
                    <div className="text-[10px] text-[#3d4a42]">Banana — precision / recall / F1 (n=94)</div>
                  </div>
                  <div className="p-2.5 rounded bg-[#fff8e1] border border-[#ffb95f]">
                    <div className="font-bold text-[#825100]">75% / 75% / 75%</div>
                    <div className="text-[10px] text-[#3d4a42]">Non-Banana — precision / recall / F1 (n=12)</div>
                  </div>
                </div>
                <p className="text-[11px] text-[#3d4a42] leading-relaxed">
                  Top-5 (i.e. trivial for 2 classes) accuracy is 100%. The weaker Non-Banana number
                  is honestly reported, not smoothed over — with only 12 test examples of that
                  class, a few misses move the percentage a lot; more negative-class data is the
                  clear next step, not a bigger model.
                </p>
              </div>
            </div>
          </div>

          <div className="lg:col-span-5 flex flex-col gap-6">
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col gap-4">
              <h3 className="text-base font-bold text-[#131b2e]">Hyperparameters &amp; Checkpoint</h3>
              <div className="flex flex-col gap-2 font-mono text-xs">
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Checkpoint Weight:</span>
                  <span className="font-bold text-[#131b2e]">tree_cls_real_v1 (tree_cls_v1.pt)</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Architecture:</span>
                  <span className="font-bold text-[#131b2e]">YOLOv8n-cls (Ultralytics, ImageNet-pretrained)</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Classes:</span>
                  <span className="font-bold text-[#131b2e]">Banana, Non_Banana</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Optimizer:</span>
                  <span className="font-bold text-[#131b2e]">SGD-momentum (chosen over AdamW, see sweep)</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Input Resolution:</span>
                  <span className="font-bold text-[#131b2e]">224 × 224 × 3 RGB</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Uncertainty Threshold:</span>
                  <span className="font-bold text-[#131b2e]">0.60 top-1 confidence</span>
                </div>
                <div className="flex justify-between py-1.5">
                  <span className="text-[#3d4a42]">Inference Latency:</span>
                  <span className="font-bold text-[#006948]">~15–25ms (CPU, this laptop)</span>
                </div>
              </div>

              <p className="text-[11px] text-[#3d4a42] leading-relaxed pt-1 border-t border-[#eaedff]">
                No GPU is involved anywhere in this deployment — the service, the database, and
                inference all run on one Windows laptop (no CUDA GPU present), reachable through a
                Cloudflare Tunnel. Latency numbers above reflect that, not a server farm.
              </p>

              <button
                type="button"
                onClick={() => onNavigate('detection-workspace')}
                className="mt-2 w-full py-2.5 px-4 rounded-xl bg-[#006948] text-white text-xs font-semibold hover:bg-[#00855d] transition-all flex items-center justify-center gap-2 cursor-pointer"
              >
                <span>Test in Workspace</span>
                <span className="material-symbols-outlined text-[16px]">arrow_forward</span>
              </button>
            </div>
          </div>

          <div className="lg:col-span-12">
            <FutureScopeSection />
          </div>
        </div>
      )}

      {/* MODEL 3 CONTENT */}
      {activeModelTab === 'model3' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          <div className="lg:col-span-7 flex flex-col gap-6">
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col gap-4">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div>
                  <span className="px-2 py-0.5 rounded bg-[#e2e7ff] text-[#006948] font-mono text-[10px] font-bold uppercase">
                    Model 3 &bull; Leaf Disease Identification
                  </span>
                  <h3 className="text-lg font-bold text-[#131b2e] mt-1">
                    Leaf Disease Identifier (U-Net, ResNet34)
                  </h3>
                </div>
                <span className="px-2.5 py-0.5 rounded-full bg-[#85f8c4] text-[#002114] font-mono text-xs font-bold">
                  96% diagnosis accuracy
                </span>
              </div>

              <div className="bg-[#131b2e] p-3 rounded-xl flex items-center justify-between text-xs font-mono text-[#85f8c4]">
                <span>POST /api/v1/leaf-disease/predict</span>
                <button
                  type="button"
                  onClick={() => copyText('curl -X POST "http://127.0.0.1:8000/api/v1/leaf-disease/predict" -H "X-Client-Id: <uuid v4>" -F "image=@leaf.jpg"', 'm3-curl')}
                  className="text-white hover:text-[#85f8c4] flex items-center gap-1 cursor-pointer text-[11px]"
                >
                  <span className="material-symbols-outlined text-[14px]">
                    {copiedApiCmd === 'm3-curl' ? 'check' : 'content_copy'}
                  </span>
                  <span>{copiedApiCmd === 'm3-curl' ? 'Copied' : 'Copy cURL'}</span>
                </button>
              </div>

              <p className="text-xs text-[#3d4a42] leading-relaxed">
                Same architecture family as Model 2 (see ADR 0019), extended to three output
                channels instead of two: <code>leaf</code>, <code>black_sigatoka</code> and{' '}
                <code>yellow_sigatoka</code>. Rather than a single whole-image label, it segments
                each disease region directly on the leaf, then derives a diagnosis (healthy, Black
                Sigatoka, or Yellow Sigatoka) from whichever disease covers the largest share of
                the leaf.
              </p>

              {/* Real data insight: label distribution */}
              <div className="flex flex-col gap-2 pt-2">
                <span className="font-mono text-xs font-bold text-[#131b2e] uppercase">
                  What the training data actually looks like
                </span>
                <p className="text-[11px] text-[#3d4a42] leading-relaxed">
                  159 labelled images (111 train / 24 val / 24 test, stratified so every class
                  appears in every split). Black Sigatoka is reasonably represented; Yellow
                  Sigatoka has only 7 images in the entire dataset &mdash; by far the hardest class
                  here, and the reason its numbers below should be read as directional rather than
                  precise.
                </p>
                <img
                  src={ASSETS.model3LabelDistribution}
                  alt="Distribution of disease labels across the training, validation and test splits"
                  className="w-full rounded-xl border border-[#dae2fd] bg-white"
                />
              </div>

              {/* Optimizer sweep */}
              <div className="flex flex-col gap-2 pt-2">
                <span className="font-mono text-xs font-bold text-[#131b2e] uppercase">
                  Optimizer selection
                </span>
                <p className="text-[11px] text-[#3d4a42] leading-relaxed">
                  AdamW and SGD were each given the same short training budget before picking one
                  for the full run, the same evidence-based check used for Models 1 and 2.
                </p>
                <img
                  src={ASSETS.model3OptimizerSweep}
                  alt="Validation Dice of AdamW vs SGD after an identical short training budget"
                  className="w-full max-w-sm mx-auto rounded-xl border border-[#dae2fd] bg-white"
                />
              </div>

              {/* Training curves */}
              <div className="flex flex-col gap-2 pt-2">
                <span className="font-mono text-xs font-bold text-[#131b2e] uppercase">
                  Training run (real, 50 epochs)
                </span>
                <img
                  src={ASSETS.model3TrainingCurves}
                  alt="Real training/validation loss and per-channel Dice curves from the three-channel U-Net training run"
                  className="w-full rounded-xl border border-[#dae2fd] bg-white"
                />
                <p className="text-[11px] text-[#3d4a42] leading-relaxed">
                  Early stopping landed on its best checkpoint at epoch 50 of an 80-epoch budget,
                  mean validation Dice 0.402. The leaf channel climbs quickly and stays high; both
                  disease channels move far more slowly and never catch up &mdash; the clearest
                  sign that more labelled disease examples, not more epochs, is what this model
                  needs next.
                </p>
              </div>

              {/* Threshold sweep */}
              <div className="flex flex-col gap-2 pt-2">
                <span className="font-mono text-xs font-bold text-[#131b2e] uppercase">
                  Picking each disease threshold honestly
                </span>
                <p className="text-[11px] text-[#3d4a42] leading-relaxed">
                  Each disease channel gets its own threshold, tuned on validation only and touched
                  against the test split exactly once, afterward, with the thresholds already
                  fixed &mdash; the same discipline as Model 2 (ADR 0012). Black Sigatoka settled
                  at 0.85, Yellow Sigatoka at 0.55; a rarer class needing a lower bar to be
                  detected at all is expected behaviour, not a bug.
                </p>
                <img
                  src={ASSETS.model3ThresholdSweep}
                  alt="IoU of each disease channel across binarization thresholds, tuned on validation only"
                  className="w-full max-w-sm mx-auto rounded-xl border border-[#dae2fd] bg-white"
                />
              </div>

              {/* Confusion matrix + real test metrics */}
              <div className="flex flex-col gap-2 pt-2">
                <span className="font-mono text-xs font-bold text-[#131b2e] uppercase">
                  Test-set results (N=24, held out, thresholds fixed beforehand)
                </span>
                <img
                  src={ASSETS.model3ConfusionMatrix}
                  alt="Image-level diagnosis confusion matrix on the held-out test set"
                  className="w-full max-w-sm mx-auto rounded-xl border border-[#dae2fd] bg-white"
                />
                <div className="grid grid-cols-3 gap-2 font-mono text-[11px] text-center pt-1">
                  <div className="p-2.5 rounded bg-[#85f8c4]/30 border border-[#85f8c4]">
                    <div className="font-bold text-[#006948]">IoU 0.914 &middot; Dice 0.955</div>
                    <div className="text-[10px] text-[#3d4a42]">Leaf channel</div>
                  </div>
                  <div className="p-2.5 rounded bg-[#fff8e1] border border-[#ffb95f]">
                    <div className="font-bold text-[#825100]">IoU 0.328 &middot; Dice 0.494</div>
                    <div className="text-[10px] text-[#3d4a42]">Black Sigatoka @ 0.85</div>
                  </div>
                  <div className="p-2.5 rounded bg-[#fff8e1] border border-[#ffb95f]">
                    <div className="font-bold text-[#825100]">IoU 0.243 &middot; Dice 0.391</div>
                    <div className="text-[10px] text-[#3d4a42]">Yellow Sigatoka @ 0.55</div>
                  </div>
                </div>
                <p className="text-[11px] text-[#3d4a42] leading-relaxed">
                  The per-pixel disease scores are modest &mdash; 159 images split across three
                  diseases doesn't give either channel much to learn from. But the image-level
                  diagnosis, which pools each mask over the whole leaf rather than demanding
                  pixel-perfect overlap, reaches 96% accuracy (23 of 24 test images correct, one
                  Yellow Sigatoka case misread). That gap is exactly why the architecture pools
                  masks into a whole-leaf verdict instead of reporting raw pixel overlap as the
                  answer.
                </p>
              </div>
            </div>
          </div>

          <div className="lg:col-span-5 flex flex-col gap-6">
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col gap-4">
              <h3 className="text-base font-bold text-[#131b2e]">Hyperparameters &amp; Checkpoint</h3>
              <div className="flex flex-col gap-2 font-mono text-xs">
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Checkpoint Weight:</span>
                  <span className="font-bold text-[#131b2e]">leaf_disease_real_v1</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Architecture:</span>
                  <span className="font-bold text-[#131b2e]">U-Net + ResNet-34 (ImageNet-pretrained)</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Channels:</span>
                  <span className="font-bold text-[#131b2e]">leaf, black_sigatoka, yellow_sigatoka</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Optimizer:</span>
                  <span className="font-bold text-[#131b2e]">AdamW (chosen over SGD, see sweep)</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Input Resolution:</span>
                  <span className="font-bold text-[#131b2e]">512 × 512 × 3 RGB</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Thresholds:</span>
                  <span className="font-bold text-[#131b2e]">0.85 black / 0.55 yellow (val-tuned)</span>
                </div>
                <div className="flex justify-between py-1.5">
                  <span className="text-[#3d4a42]">Inference Latency:</span>
                  <span className="font-bold text-[#006948]">~184ms (CPU, this laptop)</span>
                </div>
              </div>

              <p className="text-[11px] text-[#3d4a42] leading-relaxed pt-1 border-t border-[#eaedff]">
                Same laptop, same no-GPU story as Models 1 and 2: everything here runs on CPU,
                reached through a Cloudflare Tunnel, not a server farm.
              </p>

              <button
                type="button"
                onClick={() => onNavigate('leaf-detect')}
                className="mt-2 w-full py-2.5 px-4 rounded-xl bg-[#006948] text-white text-xs font-semibold hover:bg-[#00855d] transition-all flex items-center justify-center gap-2 cursor-pointer"
              >
                <span>Try it in the Leaf Workspace</span>
                <span className="material-symbols-outlined text-[16px]">arrow_forward</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* MODEL 2 CONTENT */}
      {activeModelTab === 'model2' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          <div className="lg:col-span-7 flex flex-col gap-6">
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col gap-4">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div>
                  <span className="px-2 py-0.5 rounded bg-[#e2e7ff] text-[#006948] font-mono text-[10px] font-bold uppercase">
                    Model 2 &bull; Leaf Segmentation
                  </span>
                  <h3 className="text-lg font-bold text-[#131b2e] mt-1">
                    Leaf &amp; Damage Segmenter (U-Net, ResNet34)
                  </h3>
                </div>
                <span className="px-2.5 py-0.5 rounded-full bg-[#85f8c4] text-[#002114] font-mono text-xs font-bold">
                  0.986 leaf Dice
                </span>
              </div>

              <div className="bg-[#131b2e] p-3 rounded-xl flex items-center justify-between text-xs font-mono text-[#85f8c4]">
                <span>POST /api/v1/leaf-segmentation/predict</span>
                <button
                  type="button"
                  onClick={() => copyText('curl -X POST "http://127.0.0.1:8000/api/v1/leaf-segmentation/predict" -H "X-Client-Id: <uuid v4>" -F "image=@leaf.jpg"', 'm2-curl')}
                  className="text-white hover:text-[#85f8c4] flex items-center gap-1 cursor-pointer text-[11px]"
                >
                  <span className="material-symbols-outlined text-[14px]">
                    {copiedApiCmd === 'm2-curl' ? 'check' : 'content_copy'}
                  </span>
                  <span>{copiedApiCmd === 'm2-curl' ? 'Copied' : 'Copy cURL'}</span>
                </button>
              </div>

              <p className="text-xs text-[#3d4a42] leading-relaxed">
                Two-channel segmentation: it draws the full outline of the leaf, then marks
                whatever part of it looks damaged. From those two masks it measures how much of
                the photo is leaf, how much of the leaf is affected, and how many separate lesions
                there are, then reports <code>healthy</code>, <code>affected</code> or{' '}
                <code>no_leaf</code> from those measurements rather than a learned label.
              </p>

              {/* Real data insight: pixel coverage distribution */}
              <div className="flex flex-col gap-2 pt-2">
                <span className="font-mono text-xs font-bold text-[#131b2e] uppercase">
                  What the training data actually looks like
                </span>
                <p className="text-[11px] text-[#3d4a42] leading-relaxed">
                  161 images in total: the leaf usually fills most of the frame, and damage, when
                  present, covers a small, uneven share of the leaf &mdash; exactly the kind of
                  imbalance that makes the &ldquo;affected&rdquo; channel the harder of the two to
                  learn.
                </p>
                <img
                  src={ASSETS.model2PixelCoverage}
                  alt="Distribution of leaf coverage and affected-area coverage across the training images"
                  className="w-full rounded-xl border border-[#dae2fd] bg-white"
                />
              </div>

              {/* Optimizer sweep */}
              <div className="flex flex-col gap-2 pt-2">
                <span className="font-mono text-xs font-bold text-[#131b2e] uppercase">
                  Optimizer selection
                </span>
                <p className="text-[11px] text-[#3d4a42] leading-relaxed">
                  AdamW and SGD were each given the same short training budget before picking one
                  for the full run. AdamW won here (0.611 vs 0.555 mean validation Dice after 5
                  epochs) &mdash; the opposite choice from Model 1, decided the same evidence-based way.
                </p>
                <img
                  src={ASSETS.model2OptimizerSweep}
                  alt="Validation Dice of AdamW vs SGD after an identical short training budget"
                  className="w-full max-w-sm mx-auto rounded-xl border border-[#dae2fd] bg-white"
                />
              </div>

              {/* Training curves */}
              <div className="flex flex-col gap-2 pt-2">
                <span className="font-mono text-xs font-bold text-[#131b2e] uppercase">
                  Training run (real, 62 epochs)
                </span>
                <img
                  src={ASSETS.model2TrainingCurves}
                  alt="Real training/validation loss and per-channel Dice curves from the U-Net training run"
                  className="w-full rounded-xl border border-[#dae2fd] bg-white"
                />
                <p className="text-[11px] text-[#3d4a42] leading-relaxed">
                  Early stopping (patience 15) landed on its best checkpoint at epoch 47 of an
                  80-epoch budget, mean validation Dice 0.909. The leaf channel is essentially
                  solved within the first dozen epochs; the affected channel keeps improving much
                  longer and never fully flattens out &mdash; a sign more labelled damage examples
                  would likely help more than more epochs.
                </p>
              </div>

              {/* Threshold sweep */}
              <div className="flex flex-col gap-2 pt-2">
                <span className="font-mono text-xs font-bold text-[#131b2e] uppercase">
                  Picking the damage threshold honestly
                </span>
                <p className="text-[11px] text-[#3d4a42] leading-relaxed">
                  ADR 0012 flagged that the original reference notebook tuned its threshold on the
                  test split, which is optimistic. This run tunes it on validation only (best:
                  0.90) and touches the test split exactly once, afterward, with that threshold
                  already fixed.
                </p>
                <img
                  src={ASSETS.model2ThresholdSweep}
                  alt="IoU of the affected channel across binarization thresholds, tuned on validation only"
                  className="w-full max-w-sm mx-auto rounded-xl border border-[#dae2fd] bg-white"
                />
              </div>

              {/* Confusion matrix + real test metrics */}
              <div className="flex flex-col gap-2 pt-2">
                <span className="font-mono text-xs font-bold text-[#131b2e] uppercase">
                  Test-set results (N=25, held out, threshold fixed beforehand)
                </span>
                <img
                  src={ASSETS.model2ConfusionMatrix}
                  alt="Pixel-level confusion matrix for the affected channel on the held-out test set"
                  className="w-full max-w-sm mx-auto rounded-xl border border-[#dae2fd] bg-white"
                />
                <div className="grid grid-cols-2 gap-2 font-mono text-[11px] text-center pt-1">
                  <div className="p-2.5 rounded bg-[#85f8c4]/30 border border-[#85f8c4]">
                    <div className="font-bold text-[#006948]">IoU 0.972 &middot; Dice 0.986</div>
                    <div className="text-[10px] text-[#3d4a42]">Leaf channel (precision 0.980, recall 0.991)</div>
                  </div>
                  <div className="p-2.5 rounded bg-[#fff8e1] border border-[#ffb95f]">
                    <div className="font-bold text-[#825100]">IoU 0.514 &middot; Dice 0.679</div>
                    <div className="text-[10px] text-[#3d4a42]">Affected channel @ threshold 0.90</div>
                  </div>
                </div>
                <p className="text-[11px] text-[#3d4a42] leading-relaxed">
                  The affected channel's numbers are the honest, harder story here: 161 training
                  images is not a lot of data for spotting small, irregular lesions, and that shows
                  up directly in the gap between the two channels above. Read a &ldquo;healthy&rdquo;
                  result as &ldquo;no damage above the configured threshold was found,&rdquo; not a
                  clinical guarantee.
                </p>
              </div>
            </div>
          </div>

          <div className="lg:col-span-5 flex flex-col gap-6">
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col gap-4">
              <h3 className="text-base font-bold text-[#131b2e]">Hyperparameters &amp; Checkpoint</h3>
              <div className="flex flex-col gap-2 font-mono text-xs">
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Checkpoint Weight:</span>
                  <span className="font-bold text-[#131b2e]">leaf_seg_real_v1</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Architecture:</span>
                  <span className="font-bold text-[#131b2e]">U-Net + ResNet-34 (ImageNet-pretrained)</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Channels:</span>
                  <span className="font-bold text-[#131b2e]">leaf, affected</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Optimizer:</span>
                  <span className="font-bold text-[#131b2e]">AdamW (chosen over SGD, see sweep)</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Input Resolution:</span>
                  <span className="font-bold text-[#131b2e]">512 × 512 × 3 RGB</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Affected Threshold:</span>
                  <span className="font-bold text-[#131b2e]">0.90 (validation-tuned)</span>
                </div>
                <div className="flex justify-between py-1.5">
                  <span className="text-[#3d4a42]">Inference Latency:</span>
                  <span className="font-bold text-[#006948]">~205ms (CPU, this laptop)</span>
                </div>
              </div>

              <p className="text-[11px] text-[#3d4a42] leading-relaxed pt-1 border-t border-[#eaedff]">
                Same laptop, same no-GPU story as Model 1: everything here runs on CPU, reached
                through a Cloudflare Tunnel, not a server farm.
              </p>

              <button
                type="button"
                onClick={() => onNavigate('leaf-detect')}
                className="mt-2 w-full py-2.5 px-4 rounded-xl bg-[#006948] text-white text-xs font-semibold hover:bg-[#00855d] transition-all flex items-center justify-center gap-2 cursor-pointer"
              >
                <span>Test in Leaf Workspace</span>
                <span className="material-symbols-outlined text-[16px]">arrow_forward</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* PIPELINE ARCHITECTURE CONTENT */}
      {activeModelTab === 'pipeline' && (
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col gap-6">
          <div>
            <h3 className="text-lg font-bold text-[#131b2e]">How a photo actually becomes a result</h3>
            <p className="text-xs text-[#3d4a42] mt-1 leading-relaxed">
              One FastAPI service, one laptop, no cluster. This is the real path a photo takes,
              end to end &mdash; not a scaled-up hypothetical.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 pt-2">
            <div className="p-4 rounded-xl bg-[#f2f3ff] border border-[#dae2fd] flex flex-col gap-2">
              <span className="font-mono text-[10px] text-[#006948] uppercase font-bold">
                01 &bull; Capture
              </span>
              <h4 className="text-xs font-bold text-[#131b2e]">Browser camera or upload</h4>
              <p className="text-[11px] text-[#3d4a42] leading-relaxed">
                The React frontend takes a photo (camera or file), optionally attaches GPS, and
                sends it as a normal multipart upload &mdash; no WebRTC streaming, no WebSockets.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-[#f2f3ff] border border-[#dae2fd] flex flex-col gap-2">
              <span className="font-mono text-[10px] text-[#006a61] uppercase font-bold">
                02 &bull; Tunnel &amp; API
              </span>
              <h4 className="text-xs font-bold text-[#131b2e]">Cloudflare Tunnel &rarr; FastAPI</h4>
              <p className="text-[11px] text-[#3d4a42] leading-relaxed">
                The request reaches this owner's Windows laptop through a Cloudflare Tunnel (no
                open inbound ports), where a single FastAPI/Uvicorn process handles it.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-[#f2f3ff] border border-[#dae2fd] flex flex-col gap-2">
              <span className="font-mono text-[10px] text-[#825100] uppercase font-bold">
                03 &bull; Inference
              </span>
              <h4 className="text-xs font-bold text-[#131b2e]">CPU inference, one model at a time</h4>
              <p className="text-[11px] text-[#3d4a42] leading-relaxed">
                The image is decoded, normalised and run through the requested model on CPU (no
                GPU in this deployment) inside a threadpool, so the event loop stays responsive.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-[#f2f3ff] border border-[#dae2fd] flex flex-col gap-2">
              <span className="font-mono text-[10px] text-[#ba1a1a] uppercase font-bold">
                04 &bull; Store &amp; respond
              </span>
              <h4 className="text-xs font-bold text-[#131b2e]">SQL Server + local disk</h4>
              <p className="text-[11px] text-[#3d4a42] leading-relaxed">
                The result, the original photo and a visual overlay are saved (SQL Server for
                data, local disk for images), and the API responds with the stored record.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Dataset Provenance */}
      <div className="bg-white p-6 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col gap-6">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div>
            <h3 className="text-lg font-bold text-[#131b2e]">Training Datasets</h3>
            <p className="text-xs text-[#3d4a42] mt-0.5">
              Private Roboflow exports used to train these models — not published academic
              benchmarks, so no DOI or citation is claimed for them.
            </p>
          </div>
          <button
            type="button"
            onClick={() => copyText(datasetSummary, 'dataset-summary')}
            className="px-3.5 py-1.5 rounded-lg bg-[#eaedff] text-[#131b2e] hover:bg-[#dae2fd] font-mono text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer border border-[#dae2fd]"
          >
            <span className="material-symbols-outlined text-[16px]">content_copy</span>
            <span>{copiedApiCmd === 'dataset-summary' ? 'Copied!' : 'Copy Dataset Summary'}</span>
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {benchmarkDatasets.map((ds, idx) => (
            <div
              key={idx}
              className="p-4 rounded-xl bg-[#f2f3ff] border border-[#dae2fd] flex flex-col justify-between gap-3"
            >
              <div className="flex flex-col gap-1">
                <span className="font-bold text-xs text-[#131b2e]">{ds.name}</span>
                <span className="font-mono text-[11px] text-[#006948] font-semibold">{ds.samples}</span>
                <span className="text-[11px] text-[#3d4a42]">{ds.classes}</span>
              </div>
              <div className="pt-2 border-t border-[#dae2fd] flex items-center justify-between font-mono text-[10px] text-[#3d4a42]">
                <span>{ds.resolution}</span>
                <span className="text-[#3d4a42]">{ds.source}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
