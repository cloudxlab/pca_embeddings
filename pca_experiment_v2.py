#!/usr/bin/env python3
"""
PCA Embedding Experiment v2 — Medical Domain
Expanded corpus (20 topics × 15 docs) + 5 hypotheses.

H1: Optimal PCA dim scales with number of distinct topics
H2: PCA vs Random Projection (domain-directed axes vs random reduction)
H3: PCA fit on corpus-only vs query+corpus (production realism)
H4: Hard negatives — PCA maintains separation better than raw baseline
H5: PCA benefit diminishes as corpus diversity grows (more topics → smaller gain)

Run: python3 pca_experiment_v2.py
     Requires: OPENAI_API_KEY env var
"""

import os, sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from openai import OpenAI
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

CACHE_FILE   = Path("embeddings_cache.json")
MODEL        = "text-embedding-3-small"
PCA_DIMS     = [16, 32, 48, 64, 96, 128, 192, 256]
RANDOM_SEED  = 42
rng          = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Corpus: 20 medical topics × 15 documents + 20 off-topic noise
# ---------------------------------------------------------------------------

CORPUS_LABELED = [
    # 0 — Type 2 Diabetes
    ("Metformin reduces hepatic glucose output and is the first-line drug for type 2 diabetes.", 0),
    ("GLP-1 receptor agonists like semaglutide lower blood sugar and promote weight loss in T2DM.", 0),
    ("SGLT2 inhibitors prevent renal glucose reabsorption, lowering HbA1c in diabetic patients.", 0),
    ("Insulin resistance in T2DM is driven by visceral adiposity and impaired glucose uptake.", 0),
    ("Bariatric surgery achieves T2DM remission through metabolic and incretin hormonal changes.", 0),
    ("HbA1c should be maintained below 7% in most T2DM patients to prevent microvascular complications.", 0),
    ("DPP-4 inhibitors such as sitagliptin improve glycemic control with low hypoglycemia risk.", 0),
    ("Lifestyle intervention with diet and exercise improves insulin sensitivity in prediabetes.", 0),
    ("Diabetic ketoacidosis is a life-threatening complication of uncontrolled type 1 and type 2 diabetes.", 0),
    ("Continuous glucose monitoring improves glycemic outcomes in insulin-treated diabetes patients.", 0),
    ("Pioglitazone activates PPAR-gamma receptors to reduce insulin resistance in type 2 diabetes.", 0),
    ("Cardiovascular outcomes trials show SGLT2 inhibitors reduce heart failure hospitalizations.", 0),
    ("Diabetic nephropathy is the leading cause of chronic kidney disease in developed countries.", 0),
    ("Neuropathy, retinopathy, and nephropathy are the classic microvascular complications of diabetes.", 0),
    ("Fasting plasma glucose above 126 mg/dL on two occasions confirms a diagnosis of diabetes.", 0),

    # 1 — Myocardial Infarction
    ("STEMI presents with ST-elevation, crushing chest pain, and requires urgent PCI reperfusion.", 1),
    ("Troponin I and T are highly sensitive and specific biomarkers for myocardial cell death.", 1),
    ("Aspirin and clopidogrel dual antiplatelet therapy reduces re-occlusion after acute MI.", 1),
    ("Left ventricular dysfunction after MI is assessed by echocardiography and managed with ACE inhibitors.", 1),
    ("Beta-blockers post-MI reduce mortality by decreasing heart rate and myocardial oxygen demand.", 1),
    ("NSTEMI is managed with anticoagulation, antiplatelet therapy, and early coronary angiography.", 1),
    ("Fibrinolysis with alteplase is used for STEMI when PCI cannot be performed within 120 minutes.", 1),
    ("Coronary artery disease involves atherosclerotic plaque buildup causing luminal narrowing.", 1),
    ("Statins stabilize plaques and reduce LDL cholesterol, lowering recurrent MI risk.", 1),
    ("Killip classification stratifies heart failure severity after acute myocardial infarction.", 1),
    ("Primary PCI achieves TIMI 3 flow in >90% of STEMI cases and improves survival.", 1),
    ("Mechanical complications of MI include ventricular septal defect, papillary muscle rupture.", 1),
    ("Cardiogenic shock complicating MI carries >40% in-hospital mortality despite PCI.", 1),
    ("Cardiac rehabilitation after MI improves exercise capacity, quality of life, and survival.", 1),
    ("Right ventricular infarction complicates inferior STEMI and presents with hypotension and clear lungs.", 1),

    # 2 — Antibiotic Resistance
    ("MRSA is resistant to beta-lactams via the mecA gene encoding an altered penicillin-binding protein.", 2),
    ("Vancomycin targets cell wall synthesis and remains active against most MRSA strains.", 2),
    ("Carbapenem resistance in Klebsiella pneumoniae is mediated by KPC, NDM, or OXA carbapenemases.", 2),
    ("Antibiotic stewardship programs reduce inappropriate prescribing and slow resistance emergence.", 2),
    ("Biofilm-embedded bacteria are up to 1000-fold more resistant to antibiotics than planktonic cells.", 2),
    ("Horizontal gene transfer via conjugative plasmids rapidly disseminates resistance genes.", 2),
    ("ESBL-producing E. coli cause urinary and bloodstream infections resistant to cephalosporins.", 2),
    ("Daptomycin and linezolid are alternatives for MRSA infections refractory to vancomycin.", 2),
    ("Colistin is a last-resort antibiotic for pan-drug-resistant Gram-negative infections.", 2),
    ("Clostridium difficile infection follows antibiotic disruption of the gut microbiome.", 2),
    ("Phage therapy is an experimental approach to treating antibiotic-resistant bacterial infections.", 2),
    ("Tigecycline has broad-spectrum activity against carbapenem-resistant Acinetobacter baumannii.", 2),
    ("Efflux pump overexpression confers resistance to fluoroquinolones in Pseudomonas aeruginosa.", 2),
    ("WHO classifies carbapenem-resistant Enterobacteriaceae as critical priority pathogens.", 2),
    ("Rapid diagnostic tests enable targeted antibiotic therapy and reduce broad-spectrum use.", 2),

    # 3 — Pulmonary Embolism
    ("PE classically presents with sudden dyspnea, pleuritic chest pain, and unexplained tachycardia.", 3),
    ("D-dimer is a sensitive but non-specific screening test for pulmonary embolism.", 3),
    ("CT pulmonary angiography is the gold standard imaging modality for PE diagnosis.", 3),
    ("LMWH and direct oral anticoagulants are first-line treatments for non-massive PE.", 3),
    ("Massive PE with hemodynamic instability requires systemic thrombolysis or surgical embolectomy.", 3),
    ("The Wells score stratifies pre-test probability of PE into low, moderate, and high risk.", 3),
    ("V/Q scan is used for PE diagnosis in patients with renal failure or contrast allergy.", 3),
    ("IVC filter placement is reserved for PE patients with absolute contraindication to anticoagulation.", 3),
    ("Deep vein thrombosis and PE are manifestations of venous thromboembolism (VTE).", 3),
    ("Prolonged anticoagulation for 3-6 months is recommended after provoked PE.", 3),
    ("Right heart strain on ECG (S1Q3T3 pattern) and echo suggests hemodynamically significant PE.", 3),
    ("Chronic thromboembolic pulmonary hypertension is a late complication of unresolved PE.", 3),
    ("Catheter-directed thrombolysis delivers lytic therapy directly into the pulmonary artery.", 3),
    ("Inherited thrombophilias such as Factor V Leiden increase lifetime PE risk.", 3),
    ("Pregnancy is a major risk factor for PE due to hypercoagulability and venous stasis.", 3),

    # 4 — ACE Inhibitors
    ("ACE inhibitors block angiotensin-converting enzyme, reducing angiotensin II and aldosterone.", 4),
    ("Ramipril reduces mortality in high cardiovascular risk patients in the HOPE trial.", 4),
    ("ACE inhibitor-induced cough results from bradykinin and substance P accumulation.", 4),
    ("ARBs like losartan are substituted for ACE inhibitors in patients with intolerable cough.", 4),
    ("ACE inhibitors reduce proteinuria and slow CKD progression in diabetic nephropathy.", 4),
    ("Hyperkalemia is a serious adverse effect of ACE inhibitors, especially in CKD patients.", 4),
    ("ACE inhibitors are contraindicated in bilateral renal artery stenosis due to acute kidney injury risk.", 4),
    ("Captopril was the first ACE inhibitor approved, demonstrating survival benefit in heart failure.", 4),
    ("Enalapril reduces hospitalizations and mortality in chronic systolic heart failure (CONSENSUS trial).", 4),
    ("ACE inhibitors prevent left ventricular remodeling after myocardial infarction.", 4),
    ("Angioedema is a rare but serious adverse effect of ACE inhibitors requiring immediate cessation.", 4),
    ("ACE inhibitors lower systemic vascular resistance without causing reflex tachycardia.", 4),
    ("RAAS blockade with ACE inhibitors or ARBs is first-line therapy for heart failure with reduced EF.", 4),
    ("Combination of ACE inhibitor and ARB is not recommended due to excess adverse renal effects.", 4),
    ("ACE2 receptor, distinct from ACE, is the entry point for SARS-CoV-2 into human cells.", 4),

    # 5 — Stroke
    ("Ischemic stroke is caused by thrombotic or embolic occlusion of a cerebral artery.", 5),
    ("IV alteplase within 4.5 hours of symptom onset is standard thrombolysis for ischemic stroke.", 5),
    ("Mechanical thrombectomy is effective for large vessel occlusion up to 24 hours after onset.", 5),
    ("Hypertension, atrial fibrillation, smoking, and diabetes are major modifiable stroke risk factors.", 5),
    ("Hemorrhagic stroke accounts for 15% of strokes and requires blood pressure control and reversal of anticoagulation.", 5),
    ("NIHSS score quantifies neurological deficits and guides stroke management decisions.", 5),
    ("Diffusion-weighted MRI is most sensitive for detecting acute ischemic stroke within hours.", 5),
    ("Anticoagulation with warfarin or DOAC prevents cardioembolic stroke in atrial fibrillation.", 5),
    ("Carotid endarterectomy reduces stroke risk in symptomatic patients with >70% stenosis.", 5),
    ("Penumbra salvage is the rationale for rapid reperfusion therapy in acute ischemic stroke.", 5),
    ("Dual antiplatelet therapy reduces early recurrent stroke after TIA or minor ischemic stroke.", 5),
    ("Decompressive hemicraniectomy reduces mortality in malignant MCA infarction.", 5),
    ("FAST acronym (Face, Arm, Speech, Time) helps public recognize stroke symptoms.", 5),
    ("Subarachnoid hemorrhage from berry aneurysm rupture causes sudden severe thunderclap headache.", 5),
    ("Lacunar infarcts result from small vessel disease affecting penetrating arteries in the basal ganglia.", 5),

    # 6 — COVID-19 Vaccines
    ("BNT162b2 mRNA vaccine showed 95% efficacy against symptomatic COVID-19 in phase 3 trials.", 6),
    ("mRNA vaccines deliver instructions for spike protein synthesis, inducing immune memory.", 6),
    ("Vaccine effectiveness against severe COVID-19 remains high even as neutralizing antibody titers wane.", 6),
    ("Booster doses restore neutralizing antibody levels and improve protection against Omicron.", 6),
    ("Myocarditis is a rare adverse event of mRNA vaccines, predominantly in young males after dose 2.", 6),
    ("ChAdOx1 nCoV-19 (AstraZeneca) uses an adenoviral vector to deliver spike protein antigen.", 6),
    ("VAERS and Yellow Card systems monitor post-authorization vaccine safety signals.", 6),
    ("Vaccine-induced immune thrombocytopenia (VITT) is a rare complication of adenoviral vector vaccines.", 6),
    ("Omicron BA.4/BA.5 variants show significant immune evasion from ancestral spike protein vaccines.", 6),
    ("Bivalent mRNA boosters targeting original and Omicron spike protein improve cross-variant coverage.", 6),
    ("COVID-19 vaccines reduce long COVID risk by preventing acute infection severity.", 6),
    ("mRNA vaccine lipid nanoparticles protect RNA from degradation and facilitate cellular uptake.", 6),
    ("Immunocompromised patients may require additional primary doses of COVID-19 vaccine.", 6),
    ("Vaccine hesitancy driven by misinformation is a major barrier to achieving herd immunity.", 6),
    ("Correlates of protection for COVID-19 include anti-spike IgG titers and T-cell responses.", 6),

    # 7 — Chronic Kidney Disease
    ("CKD is staged G1-G5 based on GFR and A1-A3 based on albuminuria level.", 7),
    ("ACE inhibitors and ARBs are first-line renoprotective therapy in CKD with proteinuria.", 7),
    ("Anemia of CKD arises from reduced EPO production and is treated with ESAs and IV iron.", 7),
    ("Secondary hyperparathyroidism in CKD requires phosphate binders and active vitamin D analogs.", 7),
    ("Hemodialysis three times weekly is the most common form of renal replacement therapy.", 7),
    ("SGLT2 inhibitors reduce CKD progression independent of glycemic effects (CREDENCE, DAPA-CKD trials).", 7),
    ("Peritoneal dialysis preserves residual kidney function longer than hemodialysis.", 7),
    ("Kidney transplantation offers the best outcomes for eligible end-stage kidney disease patients.", 7),
    ("Metabolic acidosis in CKD accelerates muscle catabolism and bone disease and requires bicarbonate supplementation.", 7),
    ("Fluid and dietary sodium restriction are key non-pharmacological measures in CKD management.", 7),
    ("CKD-mineral bone disorder (CKD-MBD) encompasses abnormalities in calcium, phosphate, and PTH.", 7),
    ("Cardiovascular disease is the leading cause of death in CKD patients.", 7),
    ("Renal biomarkers cystatin C and urine ACR are used for CKD staging and risk stratification.", 7),
    ("Fistula-first policy is recommended for hemodialysis access to reduce infection and thrombosis risk.", 7),
    ("Nocturnal home hemodialysis improves blood pressure, phosphate control, and quality of life.", 7),

    # 8 — Asthma
    ("Asthma is characterized by reversible airway obstruction, bronchial hyperresponsiveness, and eosinophilic inflammation.", 8),
    ("Inhaled corticosteroids reduce airway inflammation and are the cornerstone of asthma control.", 8),
    ("Short-acting beta-2 agonists like salbutamol provide rapid bronchodilation for acute symptoms.", 8),
    ("Leukotriene receptor antagonists (montelukast) are add-on therapy for mild persistent asthma.", 8),
    ("Biologic agents targeting IL-5, IgE, IL-4/IL-13, or TSLP are used for severe eosinophilic asthma.", 8),
    ("Spirometry showing FEV1/FVC ratio <0.7 with reversibility confirms obstructive airway disease.", 8),
    ("Exercise-induced bronchoconstriction is managed with pre-exercise salbutamol or LABA.", 8),
    ("Peak expiratory flow monitoring guides self-management in asthma action plans.", 8),
    ("Omalizumab targets IgE and reduces exacerbations in allergic asthma with high IgE levels.", 8),
    ("Dupilumab targets IL-4Rα and reduces exacerbations in eosinophilic asthma.", 8),
    ("Step-up and step-down therapy follows GINA guidelines based on symptom control level.", 8),
    ("Occupational asthma is caused by workplace sensitizers like isocyanates, flour, or latex.", 8),
    ("Aspirin-exacerbated respiratory disease features asthma, nasal polyps, and NSAID sensitivity.", 8),
    ("Childhood asthma is associated with atopy, allergic rhinitis, and eczema (atopic triad).", 8),
    ("Status asthmaticus is a severe acute asthma attack unresponsive to initial bronchodilator therapy.", 8),

    # 9 — Major Depressive Disorder
    ("SSRIs are first-line pharmacotherapy for major depression due to efficacy and tolerability.", 9),
    ("Cognitive behavioral therapy is as effective as antidepressants for mild-to-moderate depression.", 9),
    ("Treatment-resistant depression responds to augmentation with lithium, atypical antipsychotics, or esketamine.", 9),
    ("SNRIs like venlafaxine are effective for comorbid depression and anxiety or chronic pain.", 9),
    ("ECT remains the most effective treatment for severe refractory major depression.", 9),
    ("Psychosocial stressors, childhood trauma, and genetic factors interact to cause depression.", 9),
    ("Transcranial magnetic stimulation (TMS) is a non-invasive neuromodulation for treatment-resistant depression.", 9),
    ("The monoamine hypothesis of depression proposes that serotonin and norepinephrine deficits underlie the disorder.", 9),
    ("Esketamine nasal spray has rapid antidepressant effects mediated by NMDA receptor antagonism.", 9),
    ("Bupropion inhibits dopamine and norepinephrine reuptake and is used for atypical depression and smoking cessation.", 9),
    ("Postpartum depression affects 10-15% of mothers and requires psychotherapy or antidepressant treatment.", 9),
    ("Suicidality assessment is a critical component of the clinical evaluation of depressive disorders.", 9),
    ("Antidepressant discontinuation syndrome occurs with abrupt cessation of SSRIs or SNRIs.", 9),
    ("Bipolar depression requires careful use of mood stabilizers and avoidance of antidepressant monotherapy.", 9),
    ("Mindfulness-based cognitive therapy prevents recurrence in patients with three or more depressive episodes.", 9),

    # 10 — Parkinson's Disease
    ("Parkinson's disease is characterized by bradykinesia, rigidity, resting tremor, and postural instability.", 10),
    ("Levodopa-carbidopa is the most effective symptomatic treatment for Parkinson's motor symptoms.", 10),
    ("Alpha-synuclein aggregates into Lewy bodies, the pathological hallmark of Parkinson's disease.", 10),
    ("Deep brain stimulation of the subthalamic nucleus dramatically reduces motor fluctuations in PD.", 10),
    ("Dopamine agonists like pramipexole are used as initial therapy or adjunct to levodopa in PD.", 10),
    ("Non-motor symptoms of Parkinson's include dementia, depression, sleep disturbance, and autonomic dysfunction.", 10),
    ("MAOB inhibitors such as selegiline and rasagiline have neuroprotective and symptomatic effects in PD.", 10),
    ("LSVT LOUD and LSVT BIG are specialized speech and physical therapies for Parkinson's rehabilitation.", 10),
    ("Wearing-off phenomenon in PD results from declining dopaminergic response between levodopa doses.", 10),
    ("GBA gene mutations are the most common genetic risk factor for sporadic Parkinson's disease.", 10),
    ("Constipation and hyposmia may precede motor symptoms of Parkinson's by years (prodromal phase).", 10),
    ("Parkinson's Disease Questionnaire-39 (PDQ-39) measures quality of life in PD patients.", 10),
    ("Continuous intrajejunal levodopa infusion reduces off-time in advanced Parkinson's disease.", 10),
    ("REM sleep behavior disorder is a prodromal manifestation of synucleinopathies including PD.", 10),
    ("Neuroinflammation driven by activated microglia contributes to dopaminergic neuron loss in PD.", 10),

    # 11 — Alzheimer's Disease
    ("Alzheimer's disease is the most common cause of dementia, accounting for 60-70% of cases.", 11),
    ("Amyloid-beta plaques and neurofibrillary tau tangles are the hallmark pathological features of AD.", 11),
    ("APOE-e4 allele is the strongest genetic risk factor for late-onset Alzheimer's disease.", 11),
    ("Cholinesterase inhibitors (donepezil, rivastigmine) modestly improve cognition in Alzheimer's dementia.", 11),
    ("Lecanemab and donanemab are anti-amyloid monoclonal antibodies that slow early AD progression.", 11),
    ("The amyloid cascade hypothesis posits that amyloid-beta accumulation initiates the AD pathological cascade.", 11),
    ("PET imaging with amyloid and tau tracers allows in vivo detection of AD pathology.", 11),
    ("CSF biomarkers (Abeta42, p-tau, t-tau) support AD diagnosis before clinical dementia onset.", 11),
    ("Memantine is an NMDA receptor antagonist used for moderate-to-severe Alzheimer's disease.", 11),
    ("Neuropsychological testing including MMSE and MoCA screens for cognitive impairment in AD.", 11),
    ("Behavioral and psychological symptoms of dementia (BPSD) include agitation, psychosis, and wandering.", 11),
    ("Caregiver burden is a major challenge in dementia management and requires psychosocial support.", 11),
    ("Vascular dementia is the second most common dementia type, caused by cerebrovascular disease.", 11),
    ("Semantic memory deficits and episodic memory loss are the earliest cognitive changes in AD.", 11),
    ("Blood-based biomarkers like plasma p-tau217 show promise for non-invasive AD screening.", 11),

    # 12 — Rheumatoid Arthritis
    ("RA is a systemic autoimmune disease causing symmetric polyarthritis and joint destruction.", 12),
    ("Anti-CCP antibodies and rheumatoid factor are serological markers of RA with diagnostic value.", 12),
    ("Methotrexate is the anchor DMARD for RA, reducing inflammation and structural damage.", 12),
    ("TNF inhibitors (adalimumab, etanercept) revolutionized RA treatment for methotrexate-inadequate responders.", 12),
    ("JAK inhibitors (tofacitinib, baricitinib) are oral targeted therapies for moderate-to-severe RA.", 12),
    ("Treat-to-target strategy in RA aims for remission or low disease activity to prevent joint damage.", 12),
    ("IL-6 receptor antagonists (tocilizumab, sarilumab) suppress systemic inflammation in RA.", 12),
    ("Erosive joint disease in RA is detected by X-ray, ultrasound, or MRI of affected joints.", 12),
    ("Extra-articular RA manifestations include rheumatoid nodules, vasculitis, and interstitial lung disease.", 12),
    ("ABATACEPT blocks T-cell co-stimulation and is effective in seropositive RA.", 12),
    ("Biologic DMARDs increase infection risk, particularly tuberculosis and opportunistic infections.", 12),
    ("DAS28 and CDAI are composite disease activity scores used to monitor RA treatment response.", 12),
    ("Shared epitope HLA-DRB1 alleles confer susceptibility to seropositive RA.", 12),
    ("Cardiovascular risk is elevated twofold in RA due to systemic inflammation and traditional risk factors.", 12),
    ("Morning stiffness lasting >1 hour is a hallmark symptom distinguishing RA from osteoarthritis.", 12),

    # 13 — Sepsis
    ("Sepsis is a life-threatening organ dysfunction caused by dysregulated host response to infection.", 13),
    ("The Sepsis-3 definition uses SOFA score to identify organ dysfunction in suspected infection.", 13),
    ("Blood cultures must be obtained before administering empiric broad-spectrum antibiotics in sepsis.", 13),
    ("Septic shock is defined by vasopressor requirement and lactate >2 mmol/L despite fluid resuscitation.", 13),
    ("Norepinephrine is the preferred vasopressor for maintaining MAP >65 mmHg in septic shock.", 13),
    ("The Surviving Sepsis Campaign bundles recommend 30 mL/kg crystalloid within 3 hours of sepsis.", 13),
    ("Procalcitonin-guided antibiotic de-escalation reduces antibiotic exposure without increasing mortality.", 13),
    ("Source control — abscess drainage or infected device removal — is essential in sepsis management.", 13),
    ("Corticosteroids are recommended in vasopressor-refractory septic shock to restore hemodynamic stability.", 13),
    ("Cytokine storm in severe sepsis leads to coagulopathy, microvascular injury, and multi-organ failure.", 13),
    ("ARDS complicates sepsis in 20-30% of patients and requires lung-protective ventilation.", 13),
    ("qSOFA score (altered mentation, high RR, low BP) identifies high-risk infection patients outside ICU.", 13),
    ("Gram-negative bacteremia triggers endotoxin-mediated LPS activation of the innate immune system.", 13),
    ("Lactate clearance >10% at 2 hours predicts improved outcomes in septic shock resuscitation.", 13),
    ("Early goal-directed therapy trials (ProCESS, ARISE, ProMISe) did not show benefit over usual care.", 13),

    # 14 — Liver Cirrhosis
    ("Cirrhosis represents the end stage of hepatic fibrosis with regenerative nodules replacing normal architecture.", 14),
    ("Common causes of cirrhosis include alcohol, NASH, hepatitis B, hepatitis C, and autoimmune hepatitis.", 14),
    ("Portal hypertension in cirrhosis leads to ascites, varices, and hepatic encephalopathy.", 14),
    ("Non-selective beta-blockers reduce portal pressure and prevent variceal hemorrhage in cirrhosis.", 14),
    ("Liver transplantation is the definitive treatment for decompensated cirrhosis with MELD score >15.", 14),
    ("Spontaneous bacterial peritonitis complicates ascites and requires empiric third-generation cephalosporin.", 14),
    ("TIPS procedure decompresses portal hypertension and controls refractory ascites or variceal bleeding.", 14),
    ("Hepatic encephalopathy results from ammonia accumulation and is treated with lactulose and rifaximin.", 14),
    ("Hepatorenal syndrome is a severe complication of cirrhosis treated with terlipressin and albumin.", 14),
    ("Hepatocellular carcinoma surveillance with ultrasound every 6 months is recommended in cirrhotic patients.", 14),
    ("Child-Pugh score and MELD score stratify severity and prognosis of liver cirrhosis.", 14),
    ("Antivirals (sofosbuvir, entecavir) treat underlying viral hepatitis to halt fibrosis progression.", 14),
    ("Paracentesis with albumin replacement manages tense ascites and prevents circulatory dysfunction.", 14),
    ("Coagulopathy in cirrhosis reflects both impaired synthesis and consumption of clotting factors.", 14),
    ("Nutritional support and avoidance of alcohol are cornerstones of cirrhosis management.", 14),

    # 15 — Tuberculosis
    ("Mycobacterium tuberculosis is transmitted via respiratory droplets and causes pulmonary and extrapulmonary TB.", 15),
    ("Standard TB treatment consists of 2 months of HRZE followed by 4 months of HR (2HRZE/4HR).", 15),
    ("Latent TB infection is treated with isoniazid for 6-9 months or rifampicin-based short regimens.", 15),
    ("Multidrug-resistant TB (MDR-TB) is resistant to isoniazid and rifampicin and requires second-line agents.", 15),
    ("Bedaquiline and pretomanid are newer anti-TB drugs effective against MDR and XDR tuberculosis.", 15),
    ("HIV co-infection dramatically increases TB reactivation risk and requires integrated treatment.", 15),
    ("Tuberculin skin test and IGRA (interferon-gamma release assay) detect TB exposure and latent infection.", 15),
    ("Chest X-ray in active pulmonary TB shows upper-lobe infiltrates, cavitation, and hilar lymphadenopathy.", 15),
    ("GeneXpert MTB/RIF provides rapid TB diagnosis and rifampicin resistance detection.", 15),
    ("BCG vaccine provides protection against severe childhood TB, including meningitis and miliary TB.", 15),
    ("Drug-induced hepatotoxicity is a common adverse effect of isoniazid, rifampicin, and pyrazinamide.", 15),
    ("Directly observed therapy (DOT) improves TB treatment adherence and reduces drug resistance.", 15),
    ("Spinal tuberculosis (Pott's disease) causes vertebral collapse and potential cord compression.", 15),
    ("TB meningitis is a severe extrapulmonary form requiring prolonged treatment and dexamethasone.", 15),
    ("Contact tracing and preventive therapy are essential public health measures for TB control.", 15),

    # 16 — Breast Cancer
    ("Breast cancer is the most common malignancy in women, with invasive ductal carcinoma being predominant.", 16),
    ("Mammography screening reduces breast cancer mortality by detecting tumors at earlier, more treatable stages.", 16),
    ("BRCA1 and BRCA2 mutations confer lifetime breast cancer risk of 70% and 45% respectively.", 16),
    ("HER2-positive breast cancer is treated with trastuzumab and pertuzumab targeting HER2 receptor.", 16),
    ("Hormone receptor-positive breast cancer is treated with endocrine therapy (tamoxifen, aromatase inhibitors).", 16),
    ("Triple-negative breast cancer lacks ER, PR, and HER2 expression and has limited targeted options.", 16),
    ("CDK4/6 inhibitors (palbociclib, ribociclib) combined with endocrine therapy improve PFS in HR+ MBC.", 16),
    ("Sentinel lymph node biopsy avoids full axillary dissection and reduces lymphedema risk.", 16),
    ("Neoadjuvant chemotherapy downstages locally advanced breast cancer before surgical resection.", 16),
    ("PARP inhibitors (olaparib) are effective for BRCA-mutated metastatic breast cancer.", 16),
    ("Radiation therapy after lumpectomy reduces local recurrence to rates equivalent to mastectomy.", 16),
    ("Immunotherapy with pembrolizumab improves outcomes in high PD-L1 triple-negative breast cancer.", 16),
    ("Oncotype DX and Mammaprint genomic assays guide chemotherapy decisions in early-stage breast cancer.", 16),
    ("Bone is the most common site of metastasis in HR-positive metastatic breast cancer.", 16),
    ("Risk reduction strategies for BRCA mutation carriers include prophylactic mastectomy and oophorectomy.", 16),

    # 17 — Schizophrenia
    ("Schizophrenia is characterized by positive symptoms (hallucinations, delusions), negative symptoms, and cognitive deficits.", 17),
    ("Dopamine D2 receptor blockade is the common mechanism of all antipsychotic medications.", 17),
    ("Clozapine is the most effective antipsychotic for treatment-resistant schizophrenia.", 17),
    ("Second-generation antipsychotics have lower extrapyramidal side effects than first-generation agents.", 17),
    ("Cognitive remediation therapy and social skills training improve functional outcomes in schizophrenia.", 17),
    ("Tardive dyskinesia is a late-onset movement disorder from long-term dopamine receptor blockade.", 17),
    ("Metabolic syndrome risk (weight gain, dyslipidemia, diabetes) is elevated with second-generation antipsychotics.", 17),
    ("Early intervention in psychosis within the first episode improves long-term functional outcomes.", 17),
    ("Long-acting injectable antipsychotics improve adherence and reduce relapse rates in schizophrenia.", 17),
    ("Glutamate and NMDA receptor hypofunction is an alternative neurobiological theory of schizophrenia.", 17),
    ("Negative symptoms of schizophrenia (avolition, alogia, anhedonia) are poorly responsive to antipsychotics.", 17),
    ("Clozapine requires regular WBC monitoring due to risk of potentially fatal agranulocytosis.", 17),
    ("Cannabis use in adolescence increases the risk of psychosis and schizophrenia in vulnerable individuals.", 17),
    ("Neuroleptic malignant syndrome is a rare life-threatening reaction to antipsychotic medication.", 17),
    ("Coordinated Specialty Care programs improve outcomes in first-episode psychosis.", 17),

    # 18 — Inflammatory Bowel Disease
    ("Crohn's disease involves transmural granulomatous inflammation that can affect any part of the GI tract.", 18),
    ("Ulcerative colitis causes continuous mucosal inflammation limited to the colon and rectum.", 18),
    ("Anti-TNF biologics (infliximab, adalimumab) induce and maintain remission in moderate-to-severe IBD.", 18),
    ("Vedolizumab targets gut-selective a4b7 integrin, reducing IBD inflammation with favorable safety profile.", 18),
    ("Ustekinumab targets IL-12/23 and is effective for both Crohn's disease and ulcerative colitis.", 18),
    ("Treat-to-target in IBD aims for endoscopic remission (mucosal healing) beyond symptom control.", 18),
    ("Colonoscopy with biopsy confirms IBD diagnosis and distinguishes Crohn's from ulcerative colitis.", 18),
    ("Fecal calprotectin is a non-invasive biomarker of intestinal inflammation used to monitor IBD activity.", 18),
    ("Corticosteroids induce IBD remission but are not used for maintenance due to long-term side effects.", 18),
    ("Stricturing and penetrating complications of Crohn's often require surgical resection.", 18),
    ("Primary sclerosing cholangitis is the major hepatobiliary extraintestinal manifestation of IBD.", 18),
    ("Dysplasia surveillance colonoscopy is recommended after 8-10 years of extensive UC due to colorectal cancer risk.", 18),
    ("JAK inhibitors (tofacitinib, upadacitinib) are approved for moderate-to-severe ulcerative colitis.", 18),
    ("Nutritional therapy with exclusive enteral nutrition is an effective induction therapy for pediatric Crohn's.", 18),
    ("Ozanimod, an S1P receptor modulator, is a novel oral therapy approved for ulcerative colitis.", 18),

    # 19 — Thyroid Disorders
    ("Hypothyroidism is treated with levothyroxine, titrated to normalize TSH to the reference range.", 19),
    ("Hashimoto's thyroiditis is the most common cause of hypothyroidism in iodine-sufficient regions.", 19),
    ("Graves' disease is an autoimmune hyperthyroidism caused by TSH receptor-stimulating antibodies.", 19),
    ("Radioiodine ablation, antithyroid drugs, and thyroidectomy are treatment options for hyperthyroidism.", 19),
    ("Thyroid cancer is classified as papillary, follicular, medullary, or anaplastic based on cell origin.", 19),
    ("Fine-needle aspiration biopsy is the primary investigation for thyroid nodules with suspicious features.", 19),
    ("TSH suppression with levothyroxine reduces recurrence risk after thyroidectomy for differentiated thyroid cancer.", 19),
    ("Thyroid storm is a life-threatening thyrotoxic crisis managed with antithyroids, beta-blockers, and steroids.", 19),
    ("Myxedema coma is severe hypothyroidism with altered consciousness requiring IV T4 replacement.", 19),
    ("Ultrasound thyroid nodule characteristics (microcalcifications, irregular margins) guide biopsy decisions.", 19),
    ("Methimazole is the preferred antithyroid drug except in the first trimester of pregnancy.", 19),
    ("Central hypothyroidism from pituitary disease presents with low TSH and low free T4.", 19),
    ("Subclinical hypothyroidism with TSH 4.5-10 mIU/L in symptomatic or pregnant patients warrants treatment.", 19),
    ("BRAF V600E mutation is present in 60% of papillary thyroid cancers and drives aggressive behavior.", 19),
    ("Iodine deficiency remains the leading cause of goiter and hypothyroidism globally.", 19),

    # Off-topic noise (topic = -1)
    ("Quantum computing uses qubits exploiting superposition and entanglement for exponential speedup.", -1),
    ("The Amazon rainforest releases billions of tonnes of CO2 due to accelerated deforestation.", -1),
    ("Renaissance art emphasizes humanist perspective and naturalistic representation of the human form.", -1),
    ("Blockchain provides decentralized tamper-resistant ledgers for financial transactions.", -1),
    ("The Big Bang theory posits the universe originated from an extremely hot dense state 13.8 billion years ago.", -1),
    ("Supply chain disruptions in semiconductor manufacturing affect global electronics production.", -1),
    ("Classical music theory encompasses harmony, counterpoint, and form developed from the Baroque period.", -1),
    ("Deep-sea hydrothermal vents host chemosynthetic ecosystems independent of sunlight.", -1),
    ("Olympic sprinters achieve peak speeds above 44 km/h through explosive fast-twitch muscle activation.", -1),
    ("The French Revolution abolished the monarchy and established principles of liberty and equality.", -1),
    ("Neural networks learn hierarchical feature representations through backpropagation and gradient descent.", -1),
    ("Volcanic eruptions release sulfur dioxide, creating stratospheric aerosols that cool the climate.", -1),
    ("The International Space Station orbits at 408 km altitude and hosts continuous scientific research.", -1),
    ("Gothic architecture is characterized by pointed arches, flying buttresses, and ribbed vaulting.", -1),
    ("Coral reef bleaching is driven by elevated ocean temperatures causing expulsion of zooxanthellae.", -1),
    ("Stock options provide the right but not the obligation to buy shares at a fixed strike price.", -1),
    ("Renaissance polyphony by composers like Palestrina features interwoven independent vocal lines.", -1),
    ("Dark matter constitutes 27% of the universe but interacts only gravitationally with baryonic matter.", -1),
    ("The Silk Road connected East Asia to the Mediterranean, facilitating trade and cultural exchange.", -1),
    ("Photosynthesis converts CO2 and water into glucose using light energy captured by chlorophyll.", -1),
]

QUERIES_LABELED = [
    ("What are the pharmacological treatments for type 2 diabetes?", 0),
    ("How is myocardial infarction diagnosed and treated?", 1),
    ("How does antibiotic resistance develop and what are treatment strategies?", 2),
    ("How is pulmonary embolism diagnosed and managed?", 3),
    ("What is the mechanism and clinical use of ACE inhibitors?", 4),
    ("What are the risk factors and treatments for ischemic stroke?", 5),
    ("What is the efficacy and safety profile of COVID-19 mRNA vaccines?", 6),
    ("How is chronic kidney disease managed and what are its complications?", 7),
    ("What is the pathophysiology and treatment of asthma?", 8),
    ("What are the treatment options for major depressive disorder?", 9),
    ("What are the symptoms and treatments of Parkinson's disease?", 10),
    ("How is Alzheimer's disease diagnosed and treated?", 11),
    ("What are the treatment strategies for rheumatoid arthritis?", 12),
    ("How is sepsis defined and managed in the ICU?", 13),
    ("What are the complications and management of liver cirrhosis?", 14),
    ("How is tuberculosis diagnosed and treated?", 15),
    ("What are the treatment options for breast cancer?", 16),
    ("How is schizophrenia treated pharmacologically?", 17),
    ("What are the treatments for inflammatory bowel disease?", 18),
    ("How are thyroid disorders diagnosed and managed?", 19),
]

# Hard negatives: medically adjacent but wrong-topic texts
HARD_NEGATIVES = [
    # Looks like diabetes but is about CKD (topic 7)
    ("Insulin dose adjustment is required in stage 4-5 CKD due to reduced renal insulin clearance.", -2),
    # Looks like cardiac but is about sepsis (topic 13)
    ("Septic cardiomyopathy causes reversible myocardial depression and low ejection fraction in severe sepsis.", -2),
    # Looks like asthma but is about COPD
    ("COPD is characterized by irreversible airflow obstruction from smoking-induced emphysema and bronchitis.", -2),
    # Looks like depression but is about Alzheimer's
    ("Depression is the most common neuropsychiatric symptom of Alzheimer's disease, affecting up to 50% of patients.", -2),
    # Looks like stroke but is about MS
    ("Multiple sclerosis causes focal neurological deficits from demyelinating plaques in the CNS white matter.", -2),
    # Looks like TB but is about lung cancer
    ("Lung adenocarcinoma may present with upper-lobe mass, hilar lymphadenopathy, and hemoptysis mimicking TB.", -2),
    # Looks like RA but is about lupus
    ("SLE causes polyarthritis, malar rash, and renal involvement with ANA and anti-dsDNA antibodies.", -2),
    # Looks like CKD but is about liver cirrhosis
    ("Hyponatremia in cirrhosis results from inappropriate ADH secretion and sodium retention, not renal failure.", -2),
    # Looks like IBD but is about colorectal cancer
    ("Colorectal cancer surveillance colonoscopy is recommended every 10 years starting at age 45.", -2),
    # Looks like breast cancer but is about ovarian cancer
    ("BRCA1 mutation carriers have a 44% lifetime risk of ovarian cancer and should consider risk-reducing salpingo-oophorectomy.", -2),
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def average_precision(relevant_mask: np.ndarray) -> float:
    hits = np.cumsum(relevant_mask)
    positions = np.arange(1, len(relevant_mask) + 1)
    precisions = hits / positions
    return float(np.sum(precisions * relevant_mask) / max(1, relevant_mask.sum()))


def ndcg_at_k(relevant_mask: np.ndarray, k: int = 10) -> float:
    dcg  = sum(r / np.log2(i + 2) for i, r in enumerate(relevant_mask[:k]))
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(sorted(relevant_mask, reverse=True)[:k]))
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate(
    q_embs: np.ndarray,
    c_embs: np.ndarray,
    relevance: list[list[int]],
    k: int = 10,
) -> dict:
    q = normalize(q_embs)
    c = normalize(c_embs)
    sims = q @ c.T

    aps, ndcgs, rel_sims, irrel_sims = [], [], [], []
    for qi, rel_ids in enumerate(relevance):
        rel_set = set(rel_ids)
        ranked = np.argsort(-sims[qi])
        rel_mask = np.array([1 if r in rel_set else 0 for r in ranked])
        aps.append(average_precision(rel_mask))
        ndcgs.append(ndcg_at_k(rel_mask, k))
        for ci in range(c.shape[0]):
            (rel_sims if ci in rel_set else irrel_sims).append(sims[qi, ci])

    mr, mi = float(np.mean(rel_sims)), float(np.mean(irrel_sims))
    return {
        "MAP":     float(np.mean(aps)),
        "NDCG@10": float(np.mean(ndcgs)),
        "SimGap":  mr - mi,
        "RelSim":  mr,
        "IrrelSim": mi,
    }


def apply_pca(q: np.ndarray, c: np.ndarray, d: int, fit_on: str = "both") -> tuple:
    combined = np.vstack([q, c]) if fit_on == "both" else c
    max_comp = min(combined.shape) - 1
    if d > max_comp:
        return None, None
    pca = PCA(n_components=d, random_state=RANDOM_SEED)
    pca.fit(combined)
    return pca.transform(q), pca.transform(c)


def apply_rp(q: np.ndarray, c: np.ndarray, d: int) -> tuple:
    """Gaussian random projection to d dims."""
    n_feat = q.shape[1]
    R = rng.standard_normal((n_feat, d)) / np.sqrt(d)
    return q @ R, c @ R


def print_header(title: str):
    print(f"\n{'─'*68}")
    print(f"  {title}")
    print(f"{'─'*68}")
    print(f"  {'Dims':>6}  {'Method':<28}  {'MAP':>6}  {'NDCG@10':>8}  {'SimGap':>7}")
    print(f"  {'':─>6}  {'':─>28}  {'':─>6}  {'':─>8}  {'':─>7}")


def print_row(dims, method, metrics):
    print(f"  {dims:>6}  {method:<28}  {metrics['MAP']:>6.4f}  {metrics['NDCG@10']:>8.4f}  {metrics['SimGap']:>7.4f}")


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def get_embeddings(texts: list[str], client: OpenAI) -> np.ndarray:
    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    to_fetch = [t for t in texts if t not in cache]
    if to_fetch:
        print(f"  Fetching {len(to_fetch)} new embeddings from OpenAI...")
        for i in range(0, len(to_fetch), 100):
            batch = to_fetch[i:i+100]
            resp = client.embeddings.create(model=MODEL, input=batch)
            for item, text in zip(resp.data, batch):
                cache[text] = item.embedding
            if i + 100 < len(to_fetch):
                time.sleep(0.3)
        CACHE_FILE.write_text(json.dumps(cache))
    return np.array([cache[t] for t in texts], dtype=np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    queries_text  = [q for q, _ in QUERIES_LABELED]
    corpus_text   = [c for c, _ in CORPUS_LABELED]
    hn_text       = [c for c, _ in HARD_NEGATIVES]
    query_topics  = [t for _, t in QUERIES_LABELED]
    corpus_topics = [t for _, t in CORPUS_LABELED]

    # Relevance: query i → all corpus docs with matching topic
    relevance = [[ci for ci, ct in enumerate(corpus_topics) if ct == qt]
                 for qt in query_topics]

    n_topics  = len(set(t for t in corpus_topics if t >= 0))
    n_corpus  = len(corpus_text)
    n_noise   = sum(1 for t in corpus_topics if t == -1)
    n_total   = len(queries_text) + n_corpus

    print(f"\n{'═'*68}")
    print(f"  PCA Embedding Experiment v2 — Medical Domain (Expanded)")
    print(f"  Model: {MODEL}  |  Topics: {n_topics}  |  Docs/topic: 15")
    print(f"  Queries: {len(queries_text)}  |  Corpus: {n_corpus}  |  Noise docs: {n_noise}")
    print(f"  Total texts for embedding: {n_total}")
    print(f"{'═'*68}")

    print("\nFetching embeddings...")
    all_embs = get_embeddings(queries_text + corpus_text + hn_text, client)
    q_embs  = all_embs[:len(queries_text)]
    c_embs  = all_embs[len(queries_text):len(queries_text)+n_corpus]
    hn_embs = all_embs[len(queries_text)+n_corpus:]

    n_dims   = q_embs.shape[1]
    max_pca  = min(len(queries_text) + n_corpus, n_dims) - 1
    dims_ok  = [d for d in PCA_DIMS if d <= max_pca]

    print(f"  Embedding dims: {n_dims}  |  Max PCA components: {max_pca}")
    print(f"  PCA dims to test: {dims_ok}")

    all_results = {}

    # -----------------------------------------------------------------------
    # BASELINE
    # -----------------------------------------------------------------------
    baseline = evaluate(q_embs, c_embs, relevance)
    print_header("BASELINE — Full 1536 dimensions (no reduction)")
    print_row(n_dims, "Baseline (no PCA)", baseline)
    all_results["baseline"] = {n_dims: baseline}

    # -----------------------------------------------------------------------
    # H1: Optimal PCA dimension vs number of topics
    # Hypothesis: optimal dim lies in range [2×topics, 4×topics]
    # n_topics = 20, so we predict optimal dim ≈ 40–80
    # -----------------------------------------------------------------------
    print_header(f"H1 — Optimal PCA Dim vs Topic Count ({n_topics} topics)")
    h1_rows = {}
    for d in dims_ok:
        q_p, c_p = apply_pca(q_embs, c_embs, d, fit_on="both")
        m = evaluate(q_p, c_p, relevance)
        print_row(d, f"PCA-{d} (fit: query+corpus)", m)
        h1_rows[d] = m
    all_results["H1_pca"] = h1_rows

    best_d = max(h1_rows, key=lambda d: h1_rows[d]["MAP"])
    print(f"\n  Best PCA dim by MAP: {best_d}  "
          f"(MAP={h1_rows[best_d]['MAP']:.4f} vs baseline {baseline['MAP']:.4f})")
    ratio = best_d / n_topics
    print(f"  Ratio best_dim / n_topics = {best_d}/{n_topics} = {ratio:.1f}x")

    # -----------------------------------------------------------------------
    # H2: PCA vs Random Projection at same dims
    # Hypothesis: PCA outperforms RP — domain-directed axes matter
    # -----------------------------------------------------------------------
    print_header("H2 — PCA vs Random Projection (same dims)")
    h2_rows = {}
    for d in dims_ok:
        q_p, c_p = apply_pca(q_embs, c_embs, d, fit_on="both")
        m_pca = evaluate(q_p, c_p, relevance)

        q_r, c_r = apply_rp(q_embs, c_embs, d)
        m_rp = evaluate(q_r, c_r, relevance)

        print_row(d, f"PCA-{d}", m_pca)
        print_row(d, f"RandomProj-{d}", m_rp)
        print(f"  {'':>6}  {'  ↑ PCA advantage':<28}  {m_pca['MAP']-m_rp['MAP']:>+6.4f}  "
              f"{m_pca['NDCG@10']-m_rp['NDCG@10']:>+8.4f}  {m_pca['SimGap']-m_rp['SimGap']:>+7.4f}")
        h2_rows[d] = {"pca": m_pca, "rp": m_rp, "delta_map": m_pca["MAP"] - m_rp["MAP"]}
    all_results["H2"] = h2_rows

    pca_wins = sum(1 for v in h2_rows.values() if v["delta_map"] > 0)
    print(f"\n  PCA beats RP: {pca_wins}/{len(dims_ok)} dimensions tested")

    # -----------------------------------------------------------------------
    # H3: Fit PCA on corpus-only vs query+corpus
    # Hypothesis: corpus-only fit is comparable (more realistic in production)
    # -----------------------------------------------------------------------
    print_header("H3 — PCA Fit: corpus-only vs query+corpus")
    h3_rows = {}
    for d in dims_ok:
        q_both, c_both = apply_pca(q_embs, c_embs, d, fit_on="both")
        m_both = evaluate(q_both, c_both, relevance)

        q_corp, c_corp = apply_pca(q_embs, c_embs, d, fit_on="corpus")
        m_corp = evaluate(q_corp, c_corp, relevance)

        print_row(d, f"PCA-{d} fit=query+corpus", m_both)
        print_row(d, f"PCA-{d} fit=corpus-only", m_corp)
        print(f"  {'':>6}  {'  Δ (both − corpus-only)':<28}  "
              f"{m_both['MAP']-m_corp['MAP']:>+6.4f}  "
              f"{m_both['NDCG@10']-m_corp['NDCG@10']:>+8.4f}  "
              f"{m_both['SimGap']-m_corp['SimGap']:>+7.4f}")
        h3_rows[d] = {"both": m_both, "corpus": m_corp}
    all_results["H3"] = h3_rows

    # -----------------------------------------------------------------------
    # H4: Hard negatives — PCA vs baseline under adversarial conditions
    # Add 10 hard negatives to corpus; measure how much each method degrades
    # -----------------------------------------------------------------------
    print_header("H4 — Hard Negatives: baseline vs PCA robustness")

    # Extend corpus with hard negatives
    c_ext = np.vstack([c_embs, hn_embs])
    # Relevance unchanged (hard negatives are not relevant to any query)

    m_base_ext = evaluate(q_embs, c_ext, relevance)
    base_map_drop = baseline["MAP"] - m_base_ext["MAP"]
    print_row(n_dims, "Baseline + hard-negatives", m_base_ext)
    print(f"  {'':>6}  {'  Δ from clean baseline':<28}  {-base_map_drop:>+6.4f}")

    h4_rows = {}
    for d in dims_ok:
        q_p, c_p = apply_pca(q_embs, c_ext, d, fit_on="both")
        m_pca_ext = evaluate(q_p, c_p, relevance)
        # Compare to same-dim PCA on clean corpus
        q_clean_p, c_clean_p = apply_pca(q_embs, c_embs, d, fit_on="both")
        m_pca_clean = evaluate(q_clean_p, c_clean_p, relevance)
        drop = m_pca_clean["MAP"] - m_pca_ext["MAP"]

        print_row(d, f"PCA-{d} + hard-negatives", m_pca_ext)
        print(f"  {'':>6}  {'  Δ from clean PCA':<28}  {-drop:>+6.4f}")
        h4_rows[d] = {"clean": m_pca_clean, "hard_neg": m_pca_ext, "drop": drop}
    all_results["H4"] = h4_rows

    avg_pca_drop = np.mean([v["drop"] for v in h4_rows.values()])
    print(f"\n  Baseline MAP drop from hard negatives: {base_map_drop:+.4f}")
    print(f"  Avg PCA MAP drop from hard negatives:  {avg_pca_drop:+.4f}")
    print(f"  PCA {'more' if avg_pca_drop < base_map_drop else 'less'} robust to hard negatives")

    # -----------------------------------------------------------------------
    # H5: PCA benefit vs corpus diversity (subsamples of topics)
    # Hypothesis: fewer topics → larger PCA gain (cleaner domain axis)
    # Test with 5, 10, 15, 20 topics
    # -----------------------------------------------------------------------
    print_header("H5 — PCA Benefit vs Corpus Diversity (varying # topics)")
    d_fixed = min(32, max(dims_ok))  # fixed PCA dim for comparison

    h5_rows = []
    for n_t in [5, 10, 15, 20]:
        selected_topics = list(range(n_t))
        sel_idx = [ci for ci, ct in enumerate(corpus_topics) if ct in selected_topics]
        sel_queries = [qi for qi, qt in enumerate(query_topics) if qt in selected_topics]

        c_sub = c_embs[sel_idx]
        q_sub = q_embs[sel_queries]
        rel_sub = [[sel_idx.index(ci) for ci in relevance[qi] if ci in sel_idx]
                   for qi in sel_queries]

        m_base_sub = evaluate(q_sub, c_sub, rel_sub)

        max_comp_sub = min(q_sub.shape[0] + c_sub.shape[0], n_dims) - 1
        d_use = min(d_fixed, max_comp_sub)
        q_p, c_p = apply_pca(q_sub, c_sub, d_use, fit_on="both")
        m_pca_sub = evaluate(q_p, c_p, rel_sub)

        gain = m_pca_sub["MAP"] - m_base_sub["MAP"]
        print(f"  {n_t:>2} topics | Baseline MAP={m_base_sub['MAP']:.4f} | "
              f"PCA-{d_use} MAP={m_pca_sub['MAP']:.4f} | Gain={gain:+.4f}")
        h5_rows.append({"topics": n_t, "baseline": m_base_sub["MAP"],
                         "pca": m_pca_sub["MAP"], "gain": gain})
    all_results["H5"] = h5_rows

    gains = [r["gain"] for r in h5_rows]
    print(f"\n  Gain trend (5→20 topics): {' → '.join(f'{g:+.4f}' for g in gains)}")
    trend = "decreases" if gains[-1] < gains[0] else "increases"
    print(f"  PCA gain {trend} as corpus diversity grows  "
          f"({'H5 confirmed' if trend == 'decreases' else 'H5 not confirmed'})")

    # -----------------------------------------------------------------------
    # Variance explained summary
    # -----------------------------------------------------------------------
    print_header("Variance Explained by PCA Dimension")
    pca_diag = PCA(n_components=max(dims_ok), random_state=RANDOM_SEED)
    pca_diag.fit(np.vstack([q_embs, c_embs]))
    cumvar = np.cumsum(pca_diag.explained_variance_ratio_)
    for d in dims_ok:
        print(f"  {d:>5} dims → {cumvar[d-1]*100:5.1f}% variance  |  "
              f"compression {n_dims/d:.0f}×")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    summary_rows = []
    for d in dims_ok:
        summary_rows.append({
            "experiment": "H1_PCA",
            "dims": d,
            **{f"pca_{k}": v for k, v in all_results["H1_pca"][d].items()},
            **{f"base_{k}": v for k, v in baseline.items()},
            "map_gain_over_baseline": all_results["H1_pca"][d]["MAP"] - baseline["MAP"],
        })
    pd.DataFrame(summary_rows).to_csv("pca_results_v2.csv", index=False)
    print(f"\n  Results saved to pca_results_v2.csv\n")


if __name__ == "__main__":
    main()
