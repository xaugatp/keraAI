import leafAnalysisSpecimenImg from '../assets/images/banana_leaf_analysis_1790753633840.jpg';

export const ASSETS = {
  logo: 'https://lh3.googleusercontent.com/aida-public/AB6AXuDtNMR6qFPrbLCRmlFxcZKR9uWD45l3SbUOBGWfjQCmEn4mlPFaaF8INPgv1Zc0tpwc_0J-2tpe_bHE8tNWTHBPM793B8dz8V83DlXGoNHFUuW0GVW0sXHXC654c-v7KVXpBG-q_c-r0p4zSwrOCTwQ8bmAm6FJvSXSU9lmBKcXk6w24qjR_O4C9GaDTRXRdHs1li51sMjGyzme6C4GBjFx_qHwSX61cvga7gjoi61UdcRYoMJzRKeG',
  bananaTreeConfirmed: 'https://lh3.googleusercontent.com/aida-public/AB6AXuAbRvyb2frm-5XBAjuDOBGVz03JpJaukoca8A2xpJBod55GWlkmUFFCQUSvYc8yzAo3nlX48jve_Ft_aU2XoujCV_pZkvgWC8f2oRxTl0cWZmYOYMrfeT07oIYkBak9PQoiLA2BQHMVMC2d1cy45P-O3ew1F2x98Nb9Paar81T2iQRF0Yf4FWsSwxQZorp1yYdaWB-M2UIiFFdpaDyHwe3JeYOdAVTzbfssE_lWXyKoWr86tVgTJg_B',
  bananaFieldCalib: 'https://lh3.googleusercontent.com/aida-public/AB6AXuBdm7Qk1dHzS0nIQ36zKgbhsYdnJnebVKZYbFl2N1PjXGwGHBav8J8ryBUI_XZIxiRPU5IARRXiu49fykDxlBuq7hUuGx_zM3nJkUCQwNfKWoWk9ByMmyQUYAWvUM-viY4p5-Okf5u2MwS8kgb9u94kimqGPZRQ2GuI4POOeXruY6ylYIWRLyMdkNnedNSY_4YcNgSEE5AaESsBOnmGLD5O4yRCKvou3BqcCuqFC7rRKrU9qu6Z4Iei',
  viewfinderBanana: 'https://lh3.googleusercontent.com/aida-public/AB6AXuBXtQu3O79YKWQRngtZPQYJyh4dQdwviTv4pmT1h7iOlpWaROkgLttXe-bagwAwzNSTAUTMHpCDA5ygixYRGlKjn7eSAG5Nu3XVnvcpCJFkLAXIbxcxDEg1xO2XHzvw8bwyLTeN9T8JU8wXBkdeqBxEmvJQGj4heGeylNcXawseIEgspQD1rGruXj_YK8lpcl7PIqRU4cGIW9yriaoxexB5EnG4SuU3uSGjrfRVdPcAd_K2fueVrRvb',
  sigatokaLeafSample: 'https://lh3.googleusercontent.com/aida-public/AB6AXuDhH-9bXZLMQbi-EbqkzGcp6N8DFwIlxZLcBTwrV0bdyvsMYQqWjSkOaYTICBt3mVYn2SG4TIlITX_kUk7XzfKsC4HYMyu1wDqIeKMd3kfMO-tdUpQoxk7VkoEJgG0k5PrevrpFsCN1TLArk6QC0CneZlvKMDjF51HWLWA0ks9mkts9eQieUOMNXPvKF0X91lGFH1CVExut0YZrBpTa6IvpmBISRQXoc7DuatkdS9vnqTBllPEo-Tge',
  healthyLeafControl: 'https://lh3.googleusercontent.com/aida-public/AB6AXuChmrE0oLdUJj4jrSHq4z7WI-4eXCusjIjZsiF-Gh-BucimvzxARG7DN8ETE9o4BqKZNnfogiYSIxR7yFUHYJsu08ok6kSdaVCUKwRgZ2QV9uhyZDMI4rfuCjZ8ax6zzVSWV8KOe5ahspceTrogCjGWEdV9sQjSfM5Vkal8IgSZt1hvlheIYZqi5chMNOE2nMRymOOlp9s5y1OKJHSwPbAMAehQzt_U2wmfgcI2ARIAmEmrjMGvaauk',
  negativeHouseplant: 'https://lh3.googleusercontent.com/aida-public/AB6AXuBkhi7GQCKLsstdaYmyF-TYeebRrKrV4GpW6JUECsSDm6oJzFD7qd4v8tftUbP7V7XejUeFekwiZlK7wAo_tRCtFRtLJ6xTDBugCwje6GYXaHNCpN7MKVIEtOrGTT0AfNxcNXMa4OaE3JykJOUwxzXIvoanKpWm5dZhEWUkpLqLd0vWLaTrvPsc6dhTqAV1E9pwXNPOFFaCQsIvk3Q6m99aTYtx3Jp-kwXFQ-MwIVm27bB3ki70AlSf',
  leafAnalysisSpecimen: leafAnalysisSpecimenImg,
};

export const benchmarkDatasets = [
  {
    name: 'ICAR Musa Germplasm Foliar Dataset',
    samples: '6,420 annotated images',
    classes: 'Sigatoka complex (Black, Yellow), Cordana, Healthy Musa',
    resolution: '1920 × 1080 px RGB',
    doi: '10.1016/j.compag.2023.107892',
  },
  {
    name: 'PlantVillage Foliar Pathology Benchmark',
    samples: '4,100 field specimens',
    classes: 'Pseudocercospora, Fusarium oxysporum f. sp. cubense',
    resolution: '2048 × 1536 px RGB',
    doi: '10.3389/fpls.2022.910243',
  },
  {
    name: 'Agro-Vision Tropical Canopy Benchmark 2024',
    samples: '1,880 drone & handheld scans',
    classes: 'Multi-illumination, dew glares, edge occlusions',
    resolution: '3840 × 2160 px RGB',
    doi: '10.1038/s41598-024-58210-9',
  },
];
