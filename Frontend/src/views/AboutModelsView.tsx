import React, { useState } from 'react';
import { ViewTab } from '../types';
import { benchmarkDatasets } from '../data/mockData';

interface AboutModelsViewProps {
  onNavigate: (tab: ViewTab) => void;
}

export const AboutModelsView: React.FC<AboutModelsViewProps> = ({ onNavigate }) => {
  const [activeModelTab, setActiveModelTab] = useState<'model1' | 'model3' | 'model4' | 'pipeline'>('model1');
  const [copiedBibtex, setCopiedBibtex] = useState(false);
  const [copiedApiCmd, setCopiedApiCmd] = useState<string | null>(null);

  const bibtex = `@article{kera2024agrovision,
  title={Multi-Model Deep Learning Pipeline for Musa Taxonomy Verification, Foliar Disease Classification, and U-Net Semantic Segmentation},
  author={Faculty of Agricultural Sciences and Computer Vision Engineering Group},
  journal={Computers and Electronics in Agriculture},
  volume={214},
  pages={107892},
  year={2024},
  publisher={Elsevier}
}`;

  const copyText = (text: string, key: string) => {
    if (navigator.clipboard) {
      navigator.clipboard.writeText(text);
      if (key === 'bibtex') {
        setCopiedBibtex(true);
        setTimeout(() => setCopiedBibtex(false), 2000);
      } else {
        setCopiedApiCmd(key);
        setTimeout(() => setCopiedApiCmd(null), 2000);
      }
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
          Technical specifications, validation loss curves, neural network checkpoints, and independent FastAPI endpoints
          powering Model 1 (Taxonomy), Model 3 (Disease Classifier), and Model 4 (Semantic Segmenter).
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
          onClick={() => setActiveModelTab('model3')}
          className={`px-4 py-2.5 rounded-t-xl text-xs font-bold transition-all cursor-pointer whitespace-nowrap ${
            activeModelTab === 'model3'
              ? 'bg-white text-[#006948] border-t-2 border-[#006948] shadow-sm'
              : 'text-[#3d4a42] hover:text-[#131b2e]'
          }`}
        >
          Model 3: Leaf Disease Classifier
        </button>

        <button
          type="button"
          onClick={() => setActiveModelTab('model4')}
          className={`px-4 py-2.5 rounded-t-xl text-xs font-bold transition-all cursor-pointer whitespace-nowrap ${
            activeModelTab === 'model4'
              ? 'bg-white text-[#006948] border-t-2 border-[#006948] shadow-sm'
              : 'text-[#3d4a42] hover:text-[#131b2e]'
          }`}
        >
          Model 4: Leaf Semantic Segmenter
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
          FastAPI &amp; TensorRT Pipeline
        </button>
      </div>

      {/* MODEL 1 CONTENT */}
      {activeModelTab === 'model1' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          <div className="lg:col-span-7 flex flex-col gap-6">
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col gap-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-bold text-[#131b2e]">
                  Stage 1: Taxonomic Verification (ResNet-50-Musa-v2)
                </h3>
                <span className="px-2.5 py-0.5 rounded-full bg-[#85f8c4] text-[#002114] font-mono text-xs font-bold">
                  98.4% mAP
                </span>
              </div>
              <p className="text-xs text-[#3d4a42] leading-relaxed">
                Trained specifically to decouple <em>Musa acuminata</em> and <em>Musa balbisiana</em>{' '}
                from common tropical understory flora (e.g., Arecaceae palms, Heliconiaceae false
                bird of paradise, ornamental succulents). Minimizes false-positive calls before
                triggering heavy semantic segmentation passes.
              </p>

              {/* Loss Curve SVG Chart */}
              <div className="flex flex-col gap-2 pt-2">
                <span className="font-mono text-xs font-bold text-[#131b2e] uppercase">
                  Cross-Entropy Training &amp; Validation Loss Curves
                </span>
                <div className="w-full h-52 bg-[#f2f3ff] rounded-xl p-4 border border-[#dae2fd] relative overflow-hidden">
                  <svg className="w-full h-full" viewBox="0 0 500 160" preserveAspectRatio="none">
                    {/* Grid lines */}
                    <line x1="0" y1="40" x2="500" y2="40" stroke="#dae2fd" strokeWidth="1" strokeDasharray="3 3" />
                    <line x1="0" y1="80" x2="500" y2="80" stroke="#dae2fd" strokeWidth="1" strokeDasharray="3 3" />
                    <line x1="0" y1="120" x2="500" y2="120" stroke="#dae2fd" strokeWidth="1" strokeDasharray="3 3" />

                    {/* Training Loss Curve */}
                    <path
                      d="M 10 140 Q 80 80, 150 50 T 280 30 T 400 20 T 490 16"
                      fill="none"
                      stroke="#bccac0"
                      strokeWidth="2"
                    />
                    {/* Validation Loss Curve */}
                    <path
                      d="M 10 145 Q 80 90, 150 60 T 280 40 T 400 32 T 490 26"
                      fill="none"
                      stroke="#006948"
                      strokeWidth="2.5"
                    />
                  </svg>
                  <div className="absolute bottom-2 right-4 flex items-center gap-4 text-[10px] font-mono">
                    <span className="flex items-center gap-1 text-[#3d4a42]">
                      <span className="w-3 h-0.5 bg-[#bccac0]" /> Training Loss (0.052)
                    </span>
                    <span className="flex items-center gap-1 text-[#006948] font-bold">
                      <span className="w-3 h-0.5 bg-[#006948]" /> Val Loss (0.084)
                    </span>
                  </div>
                </div>
              </div>

              {/* Confusion Matrix Table */}
              <div className="flex flex-col gap-2 pt-2">
                <span className="font-mono text-xs font-bold text-[#131b2e] uppercase">
                  Taxonomy Confusion Matrix (Validation Set N=3,680)
                </span>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-[11px] text-center">
                  <div className="p-2.5 rounded bg-[#85f8c4]/30 border border-[#85f8c4]">
                    <div className="font-bold text-[#006948]">99.1%</div>
                    <div className="text-[10px] text-[#3d4a42]">True Musa</div>
                  </div>
                  <div className="p-2.5 rounded bg-[#f2f3ff] border border-[#dae2fd]">
                    <div className="font-bold text-[#131b2e]">0.5%</div>
                    <div className="text-[10px] text-[#3d4a42]">Musa &rarr; Palm</div>
                  </div>
                  <div className="p-2.5 rounded bg-[#f2f3ff] border border-[#dae2fd]">
                    <div className="font-bold text-[#131b2e]">0.3%</div>
                    <div className="text-[10px] text-[#3d4a42]">Musa &rarr; Heliconia</div>
                  </div>
                  <div className="p-2.5 rounded bg-[#f2f3ff] border border-[#dae2fd]">
                    <div className="font-bold text-[#131b2e]">0.1%</div>
                    <div className="text-[10px] text-[#3d4a42]">Musa &rarr; Other</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="lg:col-span-5 flex flex-col gap-6">
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col gap-4">
              <h3 className="text-base font-bold text-[#131b2e]">Hyperparameters &amp; Checkpoint</h3>
              <div className="flex flex-col gap-2 font-mono text-xs">
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Checkpoint Weight:</span>
                  <span className="font-bold text-[#131b2e]">resnet50_musa_focal_loss_e48.pt</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Backbone:</span>
                  <span className="font-bold text-[#131b2e]">ResNet-50 (Pretrained ImageNet)</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Loss Function:</span>
                  <span className="font-bold text-[#131b2e]">Focal Loss (&gamma;=2.0, &alpha;=0.25)</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Optimizer:</span>
                  <span className="font-bold text-[#131b2e]">AdamW (lr=3e-4, wd=1e-2)</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Input Resolution:</span>
                  <span className="font-bold text-[#131b2e]">224 × 224 × 3 RGB</span>
                </div>
                <div className="flex justify-between py-1.5">
                  <span className="text-[#3d4a42]">Inference Latency:</span>
                  <span className="font-bold text-[#006948]">18.4ms (FP16 TensorRT)</span>
                </div>
              </div>

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
        </div>
      )}

      {/* MODEL 3 CONTENT */}
      {activeModelTab === 'model3' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          <div className="lg:col-span-7 flex flex-col gap-6">
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col gap-4">
              <div className="flex items-center justify-between">
                <div>
                  <span className="px-2 py-0.5 rounded bg-[#e2e7ff] text-[#006948] font-mono text-[10px] font-bold uppercase">
                    Model 3 &bull; Disease Classification
                  </span>
                  <h3 className="text-lg font-bold text-[#131b2e] mt-1">
                    Banana Leaf Disease Classifier (EfficientNet-B4)
                  </h3>
                </div>
                <span className="px-2.5 py-0.5 rounded-full bg-[#85f8c4] text-[#002114] font-mono text-xs font-bold">
                  96.8% F1 Score
                </span>
              </div>

              {/* Endpoint Badge */}
              <div className="bg-[#131b2e] p-3 rounded-xl flex items-center justify-between text-xs font-mono text-[#85f8c4]">
                <span>POST /api/v1/model3/disease-classification</span>
                <button
                  type="button"
                  onClick={() => copyText('curl -X POST "https://api.keravision.ai/api/v1/model3/disease-classification" -F "file=@leaf.jpg"', 'm3-curl')}
                  className="text-white hover:text-[#85f8c4] flex items-center gap-1 cursor-pointer text-[11px]"
                >
                  <span className="material-symbols-outlined text-[14px]">
                    {copiedApiCmd === 'm3-curl' ? 'check' : 'content_copy'}
                  </span>
                  <span>{copiedApiCmd === 'm3-curl' ? 'Copied' : 'Copy cURL'}</span>
                </button>
              </div>

              <p className="text-xs text-[#3d4a42] leading-relaxed">
                Dedicated pathological classification model. Determines whether the presented leaf is healthy or unhealthy,
                diagnoses the specific foliar pathogen (Black Sigatoka, Yellow Sigatoka, or Cordana), and returns the model confidence percentage.
              </p>

              {/* Confusion Matrix Table */}
              <div className="flex flex-col gap-2 pt-2">
                <span className="font-mono text-xs font-bold text-[#131b2e] uppercase">
                  Disease Classification Validation Accuracy (N=4,100)
                </span>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-[11px] text-center">
                  <div className="p-2.5 rounded bg-[#85f8c4]/30 border border-[#85f8c4]">
                    <div className="font-bold text-[#006948]">97.4%</div>
                    <div className="text-[10px] text-[#3d4a42]">Healthy Control</div>
                  </div>
                  <div className="p-2.5 rounded bg-[#ffdad6]/40 border border-[#ba1a1a]/30">
                    <div className="font-bold text-[#ba1a1a]">96.8%</div>
                    <div className="text-[10px] text-[#3d4a42]">Black Sigatoka</div>
                  </div>
                  <div className="p-2.5 rounded bg-[#fff8e1] border border-[#ffb95f]">
                    <div className="font-bold text-[#825100]">95.1%</div>
                    <div className="text-[10px] text-[#3d4a42]">Yellow Sigatoka</div>
                  </div>
                  <div className="p-2.5 rounded bg-[#e0f2f1] border border-[#006a61]/30">
                    <div className="font-bold text-[#006a61]">96.2%</div>
                    <div className="text-[10px] text-[#3d4a42]">Cordana Spot</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="lg:col-span-5 flex flex-col gap-6">
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col gap-4">
              <h3 className="text-base font-bold text-[#131b2e]">Model 3 Specifications</h3>
              <div className="flex flex-col gap-2 font-mono text-xs">
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">API Endpoint:</span>
                  <span className="font-bold text-[#006948]">/api/v1/model3/disease-classification</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Architecture:</span>
                  <span className="font-bold text-[#131b2e]">EfficientNet-B4</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Input Resolution:</span>
                  <span className="font-bold text-[#131b2e]">380 × 380 × 3 RGB</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Loss Function:</span>
                  <span className="font-bold text-[#131b2e]">Label-Smoothed Cross-Entropy</span>
                </div>
                <div className="flex justify-between py-1.5">
                  <span className="text-[#3d4a42]">Inference Latency:</span>
                  <span className="font-bold text-[#006948]">218ms</span>
                </div>
              </div>

              <button
                type="button"
                onClick={() => onNavigate('leaf-detect')}
                className="mt-2 w-full py-2.5 px-4 rounded-xl bg-[#006948] text-white text-xs font-semibold hover:bg-[#00855d] transition-all flex items-center justify-center gap-2 cursor-pointer"
              >
                <span>Test Model 3 in Leaf Workspace</span>
                <span className="material-symbols-outlined text-[16px]">arrow_forward</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* MODEL 4 CONTENT */}
      {activeModelTab === 'model4' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          <div className="lg:col-span-7 flex flex-col gap-6">
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col gap-4">
              <div className="flex items-center justify-between">
                <div>
                  <span className="px-2 py-0.5 rounded bg-[#e2e7ff] text-[#006948] font-mono text-[10px] font-bold uppercase">
                    Model 4 &bull; Semantic Segmentation
                  </span>
                  <h3 className="text-lg font-bold text-[#131b2e] mt-1">
                    Leaf Lesion Semantic Segmenter (U-Net)
                  </h3>
                </div>
                <span className="px-2.5 py-0.5 rounded-full bg-[#89f5e7] text-[#00201d] font-mono text-xs font-bold">
                  0.912 mDice
                </span>
              </div>

              {/* Endpoint Badge */}
              <div className="bg-[#131b2e] p-3 rounded-xl flex items-center justify-between text-xs font-mono text-[#85f8c4]">
                <span>POST /api/v1/model4/semantic-segmentation</span>
                <button
                  type="button"
                  onClick={() => copyText('curl -X POST "https://api.keravision.ai/api/v1/model4/semantic-segmentation" -F "file=@leaf.jpg"', 'm4-curl')}
                  className="text-white hover:text-[#85f8c4] flex items-center gap-1 cursor-pointer text-[11px]"
                >
                  <span className="material-symbols-outlined text-[14px]">
                    {copiedApiCmd === 'm4-curl' ? 'check' : 'content_copy'}
                  </span>
                  <span>{copiedApiCmd === 'm4-curl' ? 'Copied' : 'Copy cURL'}</span>
                </button>
              </div>

              <p className="text-xs text-[#3d4a42] leading-relaxed">
                Dedicated semantic segmentation model. Isolate infected regions directly on the leaf image,
                differentiates between active necrotic lesions and surrounding chlorotic halos, and computes the exact percentage of infected leaf area.
              </p>

              {/* Dice Progression SVG */}
              <div className="flex flex-col gap-2 pt-2">
                <span className="font-mono text-xs font-bold text-[#131b2e] uppercase">
                  Multi-Class Mean Dice Metric Progression (60 Epochs)
                </span>
                <div className="w-full h-52 bg-[#f2f3ff] rounded-xl p-4 border border-[#dae2fd] relative overflow-hidden">
                  <svg className="w-full h-full" viewBox="0 0 500 160" preserveAspectRatio="none">
                    <line x1="0" y1="40" x2="500" y2="40" stroke="#dae2fd" strokeWidth="1" strokeDasharray="3 3" />
                    <line x1="0" y1="80" x2="500" y2="80" stroke="#dae2fd" strokeWidth="1" strokeDasharray="3 3" />
                    <line x1="0" y1="120" x2="500" y2="120" stroke="#dae2fd" strokeWidth="1" strokeDasharray="3 3" />

                    {/* Healthy Tissue Dice */}
                    <path
                      d="M 10 130 Q 100 40, 200 25 T 350 18 T 490 14"
                      fill="none"
                      stroke="#006948"
                      strokeWidth="2.5"
                    />
                    {/* Necrotic Core Dice */}
                    <path
                      d="M 10 150 Q 120 70, 220 40 T 360 30 T 490 25"
                      fill="none"
                      stroke="#ba1a1a"
                      strokeWidth="2"
                    />
                    {/* Chlorosis Halo Dice */}
                    <path
                      d="M 10 155 Q 120 90, 220 60 T 360 45 T 490 38"
                      fill="none"
                      stroke="#ffb95f"
                      strokeWidth="2"
                    />
                  </svg>
                  <div className="absolute bottom-2 right-4 flex items-center gap-4 text-[10px] font-mono">
                    <span className="flex items-center gap-1 text-[#006948] font-bold">
                      <span className="w-3 h-0.5 bg-[#006948]" /> Healthy (0.965)
                    </span>
                    <span className="flex items-center gap-1 text-[#ba1a1a] font-bold">
                      <span className="w-3 h-0.5 bg-[#ba1a1a]" /> Necrotic (0.912)
                    </span>
                    <span className="flex items-center gap-1 text-[#825100] font-bold">
                      <span className="w-3 h-0.5 bg-[#ffb95f]" /> Chlorosis (0.884)
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="lg:col-span-5 flex flex-col gap-6">
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col gap-4">
              <h3 className="text-base font-bold text-[#131b2e]">Model 4 Specifications</h3>
              <div className="flex flex-col gap-2 font-mono text-xs">
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">API Endpoint:</span>
                  <span className="font-bold text-[#006948]">/api/v1/model4/semantic-segmentation</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Architecture:</span>
                  <span className="font-bold text-[#131b2e]">U-Net + ResNet-34 Feature Pyramid</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Loss Formulation:</span>
                  <span className="font-bold text-[#131b2e]">Combined Dice + Lovász-Softmax</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-[#eaedff]">
                  <span className="text-[#3d4a42]">Input Resolution:</span>
                  <span className="font-bold text-[#131b2e]">512 × 512 × 3 RGB</span>
                </div>
                <div className="flex justify-between py-1.5">
                  <span className="text-[#3d4a42]">Inference Latency:</span>
                  <span className="font-bold text-[#006948]">364ms</span>
                </div>
              </div>

              <button
                type="button"
                onClick={() => onNavigate('leaf-detect')}
                className="mt-2 w-full py-2.5 px-4 rounded-xl bg-[#006948] text-white text-xs font-semibold hover:bg-[#00855d] transition-all flex items-center justify-center gap-2 cursor-pointer"
              >
                <span>Test Model 4 in Leaf Workspace</span>
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
            <h3 className="text-lg font-bold text-[#131b2e]">
              FastAPI Inference Engine &amp; Asynchronous TensorRT Dispatch
            </h3>
            <p className="text-xs text-[#3d4a42] mt-1 leading-relaxed">
              High-throughput microservice architecture built to support real-time field camera
              streams over WebSockets and high-resolution batch orthomosaics over REST OpenAPI.
            </p>
          </div>

          {/* Architecture Diagram */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 pt-2">
            <div className="p-4 rounded-xl bg-[#f2f3ff] border border-[#dae2fd] flex flex-col gap-2">
              <span className="font-mono text-[10px] text-[#006948] uppercase font-bold">
                01 • Ingestion Layer
              </span>
              <h4 className="text-xs font-bold text-[#131b2e]">Edge Client &amp; Field Camera</h4>
              <p className="text-[11px] text-[#3d4a42] leading-relaxed">
                Mobile Safari/Chrome WebRTC video frame capture, normalized client-side to Float32 RGB.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-[#f2f3ff] border border-[#dae2fd] flex flex-col gap-2">
              <span className="font-mono text-[10px] text-[#006a61] uppercase font-bold">
                02 • Dispatch Broker
              </span>
              <h4 className="text-xs font-bold text-[#131b2e]">FastAPI Async Router</h4>
              <p className="text-[11px] text-[#3d4a42] leading-relaxed">
                ASGI event loop with non-blocking WebSocket callbacks and Redis task queue isolation.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-[#f2f3ff] border border-[#dae2fd] flex flex-col gap-2">
              <span className="font-mono text-[10px] text-[#825100] uppercase font-bold">
                03 • Hardware Acceleration
              </span>
              <h4 className="text-xs font-bold text-[#131b2e]">TensorRT &amp; CUDA Workers</h4>
              <p className="text-[11px] text-[#3d4a42] leading-relaxed">
                Quantized INT8/FP16 serialized engine running on NVIDIA T4/A100 server clusters.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-[#f2f3ff] border border-[#dae2fd] flex flex-col gap-2">
              <span className="font-mono text-[10px] text-[#ba1a1a] uppercase font-bold">
                04 • Agronomic Synthesis
              </span>
              <h4 className="text-xs font-bold text-[#131b2e]">Polygon Mask &amp; PSI Index</h4>
              <p className="text-[11px] text-[#3d4a42] leading-relaxed">
                Contour extraction, damage ratio calculation, and prescriptive agronomic dosage output.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Dataset Provenance & Benchmark Citations */}
      <div className="bg-white p-6 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col gap-6">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-bold text-[#131b2e]">Training Datasets &amp; Benchmarks</h3>
            <p className="text-xs text-[#3d4a42] mt-0.5">
              Curated corpora annotated by tropical plant pathologists and computer vision researchers.
            </p>
          </div>
          <button
            type="button"
            onClick={() => copyText(bibtex, 'bibtex')}
            className="px-3.5 py-1.5 rounded-lg bg-[#eaedff] text-[#131b2e] hover:bg-[#dae2fd] font-mono text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer border border-[#dae2fd]"
          >
            <span className="material-symbols-outlined text-[16px]">content_copy</span>
            <span>{copiedBibtex ? 'BibTeX Copied!' : 'Copy BibTeX Citation'}</span>
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
                <span className="text-[#006948] underline">DOI: {ds.doi}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
