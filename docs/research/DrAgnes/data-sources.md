# DrAgnes Data Sources

**Status**: Research & Planning
**Date**: 2026-03-21

## Overview

DrAgnes requires diverse, high-quality dermoscopic imaging data for training, validation, and ongoing enrichment. This document catalogs available datasets, medical literature sources, and real-world data streams that will feed the platform.

## Training Datasets

### 1. HAM10000 (Human Against Machine with 10,000 training images)

- **Source**: Medical University of Vienna / ViDIR Group
- **Size**: 10,015 dermoscopic images
- **Classes**: 7 lesion types
  - Actinic keratosis / Bowen's disease (akiec): 327 images
  - Basal cell carcinoma (bcc): 514 images
  - Benign keratosis (bkl): 1,099 images
  - Dermatofibroma (df): 115 images
  - Melanoma (mel): 1,113 images
  - Melanocytic nevus (nv): 6,705 images
  - Vascular lesion (vasc): 142 images
- **Resolution**: Variable (typically 600x450)
- **Ground Truth**: Histopathologic confirmation for ~50%, expert consensus for remainder
- **License**: CC BY-NC-SA 4.0
- **Use Case**: Primary training dataset for initial 7-class model
- **Citation**: Tschandl, P., Rosendahl, C. & Kittler, H. (2018). The HAM10000 dataset.

**Key Considerations**:
- Heavy class imbalance (67% melanocytic nevi). Requires oversampling (SMOTE or augmentation) for minority classes.
- Limited Fitzpatrick V-VI representation. Must supplement with diverse skin tone datasets.
- Non-standardized imaging conditions. Preprocessing pipeline must handle heterogeneous inputs.

### 2. ISIC Archive (International Skin Imaging Collaboration)

- **Source**: ISIC / Memorial Sloan Kettering Cancer Center
- **Size**: 70,000+ images (2024 archive)
- **Classes**: Extended taxonomy (25+ lesion types in later challenges)
- **Challenges**: ISIC 2016, 2017, 2018, 2019, 2020 -- each with labeled competition data
- **Resolution**: Variable (up to 4000x3000)
- **Ground Truth**: Mix of histopathology and expert annotation
- **License**: CC BY-NC 4.0 (varies by year)
- **Use Case**: Extended training, validation, benchmarking against ISIC challenge leaderboards

**Key Subsets**:
| Year | Images | Task |
|------|--------|------|
| ISIC 2016 | 1,279 | Binary (melanoma vs. benign) |
| ISIC 2017 | 2,750 | 3-class (melanoma, seborrheic keratosis, benign nevus) |
| ISIC 2018 | 10,015 | 7-class (same as HAM10000) |
| ISIC 2019 | 25,331 | 8-class (added squamous cell carcinoma) |
| ISIC 2020 | 33,126 | Binary (melanoma vs. benign) with metadata |

### 3. BCN20000 (Barcelona Dermoscopy Dataset)

- **Source**: Hospital Clinic de Barcelona
- **Size**: 19,424 dermoscopic images
- **Classes**: 8 diagnostic categories
- **Resolution**: Standardized at 1024x768
- **Ground Truth**: Histopathologic confirmation
- **License**: Research use (requires data use agreement)
- **Use Case**: European population diversity, high-quality histopathology labels

**Distinctive Features**:
- All images from a single institutional dermoscopy unit (consistent quality)
- Higher proportion of actinic keratoses and SCCs than HAM10000
- Includes patient metadata (age, sex, body site)
- Mediterranean population demographics

### 4. PH2 Dataset

- **Source**: University of Porto / ADDI project
- **Size**: 200 dermoscopic images
- **Classes**: 3 types
  - Common nevi: 80 images
  - Atypical nevi: 80 images
  - Melanoma: 40 images
- **Resolution**: 768x560 (8-bit RGB)
- **Ground Truth**: Expert dermatologist annotation + medical consensus
- **Annotations**: Manual segmentation masks, dermoscopic features (colors, structures, symmetry)
- **License**: Academic research use
- **Use Case**: Rich dermoscopic feature annotation for ABCDE/7-point validation

**Unique Value**: Each image includes expert-annotated dermoscopic structures (globules, streaks, blue-white veil, regression structures, dots). This enables training of the ABCDE and 7-point checklist modules, not just the CNN classifier.

### 5. Derm7pt Dataset

- **Source**: Simon Fraser University / University of British Columbia
- **Size**: 1,011 cases
- **Content**: Paired clinical + dermoscopic images for each case
- **Classes**: Melanoma vs. non-melanoma (binary) + 7-point checklist criteria
- **Annotations**: Full 7-point checklist scoring by experts
  - Atypical pigment network
  - Blue-whitish veil
  - Atypical vascular pattern
  - Irregular streaks
  - Irregular dots/globules
  - Irregular blotches
  - Regression structures
- **License**: Research use
- **Use Case**: Training the 7-point checklist automation module; validating multi-image (clinical+dermoscopic) analysis

### 6. DERMNET (Dermoscopy Image Archive)

- **Source**: DermNet NZ (New Zealand Dermatological Society)
- **Size**: 23,000+ images across 600+ skin conditions
- **Content**: Clinical photographs (not dermoscopic) with expert descriptions
- **License**: Non-commercial educational use
- **Use Case**: Clinical photo training for non-dermoscopic input mode; educational reference

### 7. Fitzpatrick17k Dataset

- **Source**: Stanford Medicine / DDI (Diverse Dermatology Images)
- **Size**: 16,577 clinical images
- **Content**: 114 skin conditions with Fitzpatrick skin type labels (I-VI)
- **Key Feature**: Explicit skin tone diversity labeling
- **License**: Research use
- **Use Case**: Bias evaluation and mitigation. Ensuring DrAgnes performs equally across all skin types.

**Critical for Equity**: Most existing dermatology AI systems show degraded performance on darker skin tones (Fitzpatrick V-VI). The Fitzpatrick17k dataset enables stratified evaluation to ensure DrAgnes does not perpetuate this bias.

### 8. PAD-UFES-20

- **Source**: Federal University of Espirito Santo (Brazil)
- **Size**: 2,298 images across 6 skin lesion types
- **Content**: Smartphone-captured clinical images (not dermoscopic)
- **Key Feature**: Real-world smartphone capture conditions (not clinical photography)
- **License**: CC BY 4.0
- **Use Case**: Validating performance with non-DermLite smartphone images; accessibility for resource-limited settings

## Medical Literature Sources

### 9. PubMed / MEDLINE

- **Access**: pi.ruv.io brain PubMed integration (`crates/mcp-brain-server/src/pubmed.rs`)
- **Content**: 36 million+ biomedical citations
- **Use Cases**:
  - Automated literature review for new lesion findings
  - Evidence enrichment for diagnostic suggestions
  - Treatment guideline updates
  - Epidemiological context for risk assessment
- **Integration**: Brain `brain_page_evidence` API attaches PubMed references to DrAgnes findings

**Key Search Strategies**:
```
"dermoscopy" AND "melanoma" AND "deep learning"
"skin lesion classification" AND "convolutional neural network"
"dermoscopic features" AND "machine learning"
"skin cancer" AND "mobile health" AND "telemedicine"
"dermatology" AND "artificial intelligence" AND "clinical validation"
"Fitzpatrick skin type" AND "algorithmic bias"
```

### 10. AAD Clinical Guidelines

- **Source**: American Academy of Dermatology
- **Content**: Evidence-based guidelines for skin cancer screening, diagnosis, and management
- **Key Guidelines**:
  - Melanoma: Clinical practice guidelines for diagnosis and management
  - Nonmelanoma skin cancer: Basal cell and squamous cell carcinoma
  - Skin cancer prevention and early detection
  - Dermoscopy standards and training
- **Use Case**: Codifying clinical decision rules into DrAgnes recommendation engine

### 11. British Association of Dermatologists (BAD) Guidelines

- **Source**: BAD
- **Content**: UK-based clinical guidelines complementing AAD
- **Key Difference**: Greater emphasis on teledermatology pathways
- **Use Case**: International clinical standard reference; teledermatology workflow design

## Regulatory & Safety Data Sources

### 12. FDA MAUDE Database

- **Source**: FDA Manufacturer and User Facility Device Experience Database
- **Content**: Adverse event reports for medical devices
- **Search Terms**: Dermatoscope, dermoscopy, DermLite, skin imaging, AI dermatology
- **Use Case**: Post-market surveillance for DermLite devices; safety signal detection for AI dermatology tools
- **Integration**: Periodic automated queries via FDA openFDA API

### 13. ClinicalTrials.gov

- **Source**: US National Library of Medicine
- **Content**: Registry of clinical studies
- **Active Dermatology AI Trials** (as of 2026):
  - AI-assisted melanoma screening in primary care
  - Deep learning for dermoscopic pattern analysis
  - Smartphone-based skin cancer detection validation
  - Teledermatology with AI triage
- **Use Case**: Monitoring competitive landscape; identifying validation study opportunities

### 14. SEER (Surveillance, Epidemiology, and End Results)

- **Source**: National Cancer Institute
- **Content**: Cancer incidence and survival data from US population registries
- **Key Data**:
  - Melanoma incidence by age, sex, race, anatomic site
  - Stage at diagnosis distribution
  - Survival rates by stage and treatment
  - Temporal trends (1975-present)
- **Use Case**: Population-level risk calibration; prevalence priors for Bayesian classification; outcome validation

### 15. GBD (Global Burden of Disease)

- **Source**: Institute for Health Metrics and Evaluation (IHME)
- **Content**: Global epidemiological data for 369 diseases across 204 countries
- **Use Case**: International deployment planning; understanding regional lesion distribution differences

## Real-World Data Streams (Post-Deployment)

### 16. Practice Contributions (via Brain)

- **Source**: DrAgnes-participating practices
- **Content**: De-identified embeddings, classification results, clinician feedback
- **Volume Projection**: 100-1,000 contributions/day at scale
- **Privacy**: All contributions go through the PII stripping and DP pipeline
- **Use Case**: Continuous model improvement; population-level insights

### 17. DermLite Device Telemetry

- **Source**: DermLite devices (with user consent)
- **Content**: Device model, capture settings, image quality metrics (no images)
- **Use Case**: Optimizing preprocessing for specific device models; quality assurance

### 18. EHR Integration Data (Future)

- **Source**: Epic FHIR, Cerner, athenahealth APIs
- **Content**: De-identified diagnosis codes (ICD-10), procedure codes, pathology reports
- **Privacy**: FHIR Bulk Data with patient consent; de-identified before analytics
- **Use Case**: Ground truth validation via histopathology; outcome tracking

## Dataset Preparation Pipeline

```
Raw Dataset
    │
    ▼
Quality Filtering
    ├── Remove duplicates (perceptual hashing)
    ├── Remove low-quality images (blur detection, exposure check)
    ├── Verify label consistency (multi-expert consensus)
    └── Flag ambiguous cases for expert review
    │
    ▼
Standardization
    ├── Resize to 224x224 (bilinear, maintaining aspect ratio with padding)
    ├── Color normalization (Shades of Gray algorithm)
    ├── Hair removal (DullRazor)
    ├── Lesion segmentation (for feature extraction)
    └── ImageNet normalization (mean/std)
    │
    ▼
Augmentation (for minority classes)
    ├── Random rotation (0-360 degrees)
    ├── Random horizontal/vertical flip
    ├── Random brightness/contrast adjustment (+/- 20%)
    ├── Random elastic deformation
    ├── Cutout / random erasing
    └── Mixup (alpha=0.2) between same-class samples
    │
    ▼
Split Strategy
    ├── Train: 70% (stratified by class and Fitzpatrick type)
    ├── Validation: 15% (stratified)
    ├── Test: 15% (stratified, held out completely)
    └── Note: Patient-level splitting (no image from same lesion in multiple sets)
    │
    ▼
Embedding Generation
    ├── ruvector-cnn MobileNetV3 Small → 576-dim embeddings
    ├── RlmEmbedder projection → 128-dim for HNSW
    ├── PiQ3 quantization for compressed search
    └── Store in brain as reference vectors
```

## Data Governance

### Data Use Agreements

| Dataset | Agreement Type | Restrictions |
|---------|---------------|-------------|
| HAM10000 | CC BY-NC-SA 4.0 | Non-commercial, share-alike, attribution |
| ISIC Archive | CC BY-NC 4.0 | Non-commercial, attribution |
| BCN20000 | Institutional DUA | Research use only; requires ethics approval |
| PH2 | Academic DUA | Academic research only |
| Derm7pt | Academic DUA | Research use only |
| Fitzpatrick17k | Research DUA | Research use; fairness evaluation |
| PAD-UFES-20 | CC BY 4.0 | Attribution only (most permissive) |

### Commercial Licensing Considerations

For commercial deployment of DrAgnes, only CC BY 4.0 and public domain datasets can be used without licensing negotiation. Commercial licensing or data use agreements must be obtained for:
- HAM10000 (CC BY-NC-SA -- non-commercial restriction)
- ISIC Archive (CC BY-NC -- non-commercial restriction)
- BCN20000 (institutional agreement required)

**Alternative**: Train on CC BY 4.0 datasets and practice-contributed data only. The brain's collective learning mechanism means the model improves from real-world use regardless of initial training data license.

### Ethical Considerations

1. **Representation**: Actively seek datasets with Fitzpatrick V-VI representation to prevent bias
2. **Consent**: All practice-contributed data requires patient consent (opt-in, not opt-out)
3. **Transparency**: Publish model cards documenting training data composition, known limitations, and performance by subgroup
4. **Feedback loops**: Monitor for disparate impact in production; retrain if bias detected
5. **Data sovereignty**: Respect regional data handling requirements (GDPR data residency, etc.)
