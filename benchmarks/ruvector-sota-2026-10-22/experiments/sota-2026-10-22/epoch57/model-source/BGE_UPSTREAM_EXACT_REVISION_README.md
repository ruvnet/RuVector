---
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- transformers
- mteb
model-index:
- name: bge-small-en-v1.5
  results:
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_counterfactual
      name: MTEB AmazonCounterfactualClassification (en)
      config: en
      split: test
      revision: e8379541af4e31359cca9fbcf4b00f2671dba205
    metrics:
    - type: accuracy
      value: 73.79104477611939
    - type: ap
      value: 37.21923821573361
    - type: f1
      value: 68.0914945617093
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_polarity
      name: MTEB AmazonPolarityClassification
      config: default
      split: test
      revision: e2d317d38cd51312af73b3d32a06d1a08b442046
    metrics:
    - type: accuracy
      value: 92.75377499999999
    - type: ap
      value: 89.46766124546022
    - type: f1
      value: 92.73884001331487
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_reviews_multi
      name: MTEB AmazonReviewsClassification (en)
      config: en
      split: test
      revision: 1399c76144fd37290681b995c656ef9b2e06e26d
    metrics:
    - type: accuracy
      value: 46.986
    - type: f1
      value: 46.55936786727896
  - task:
      type: Retrieval
    dataset:
      type: arguana
      name: MTEB ArguAna
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 35.846000000000004
    - type: map_at_10
      value: 51.388
    - type: map_at_100
      value: 52.132999999999996
    - type: map_at_1000
      value: 52.141000000000005
    - type: map_at_3
      value: 47.037
    - type: map_at_5
      value: 49.579
    - type: mrr_at_1
      value: 36.558
    - type: mrr_at_10
      value: 51.658
    - type: mrr_at_100
      value: 52.402
    - type: mrr_at_1000
      value: 52.410000000000004
    - type: mrr_at_3
      value: 47.345
    - type: mrr_at_5
      value: 49.797999999999995
    - type: ndcg_at_1
      value: 35.846000000000004
    - type: ndcg_at_10
      value: 59.550000000000004
    - type: ndcg_at_100
      value: 62.596
    - type: ndcg_at_1000
      value: 62.759
    - type: ndcg_at_3
      value: 50.666999999999994
    - type: ndcg_at_5
      value: 55.228
    - type: precision_at_1
      value: 35.846000000000004
    - type: precision_at_10
      value: 8.542
    - type: precision_at_100
      value: 0.984
    - type: precision_at_1000
      value: 0.1
    - type: precision_at_3
      value: 20.389
    - type: precision_at_5
      value: 14.438
    - type: recall_at_1
      value: 35.846000000000004
    - type: recall_at_10
      value: 85.42
    - type: recall_at_100
      value: 98.43499999999999
    - type: recall_at_1000
      value: 99.644
    - type: recall_at_3
      value: 61.166
    - type: recall_at_5
      value: 72.191
  - task:
      type: Clustering
    dataset:
      type: mteb/arxiv-clustering-p2p
      name: MTEB ArxivClusteringP2P
      config: default
      split: test
      revision: a122ad7f3f0291bf49cc6f4d32aa80929df69d5d
    metrics:
    - type: v_measure
      value: 47.402770198163594
  - task:
      type: Clustering
    dataset:
      type: mteb/arxiv-clustering-s2s
      name: MTEB ArxivClusteringS2S
      config: default
      split: test
      revision: f910caf1a6075f7329cdf8c1a6135696f37dbd53
    metrics:
    - type: v_measure
      value: 40.01545436974177
  - task:
      type: Reranking
    dataset:
      type: mteb/askubuntudupquestions-reranking
      name: MTEB AskUbuntuDupQuestions
      config: default
      split: test
      revision: 2000358ca161889fa9c082cb41daa8dcfb161a54
    metrics:
    - type: map
      value: 62.586465273207196
    - type: mrr
      value: 74.42169019038825
  - task:
      type: STS
    dataset:
      type: mteb/biosses-sts
      name: MTEB BIOSSES
      config: default
      split: test
      revision: d3fb88f8f02e40887cd149695127462bbcf29b4a
    metrics:
    - type: cos_sim_pearson
      value: 85.1891186537969
    - type: cos_sim_spearman
      value: 83.75492046087288
    - type: euclidean_pearson
      value: 84.11766204805357
    - type: euclidean_spearman
      value: 84.01456493126516
    - type: manhattan_pearson
      value: 84.2132950502772
    - type: manhattan_spearman
      value: 83.89227298813377
  - task:
      type: Classification
    dataset:
      type: mteb/banking77
      name: MTEB Banking77Classification
      config: default
      split: test
      revision: 0fd18e25b25c072e09e0d92ab615fda904d66300
    metrics:
    - type: accuracy
      value: 85.74025974025975
    - type: f1
      value: 85.71493566466381
  - task:
      type: Clustering
    dataset:
      type: mteb/biorxiv-clustering-p2p
      name: MTEB BiorxivClusteringP2P
      config: default
      split: test
      revision: 65b79d1d13f80053f67aca9498d9402c2d9f1f40
    metrics:
    - type: v_measure
      value: 38.467181385006434
  - task:
      type: Clustering
    dataset:
      type: mteb/biorxiv-clustering-s2s
      name: MTEB BiorxivClusteringS2S
      config: default
      split: test
      revision: 258694dd0231531bc1fd9de6ceb52a0853c6d908
    metrics:
    - type: v_measure
      value: 34.719496037339056
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackAndroidRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 29.587000000000003
    - type: map_at_10
      value: 41.114
    - type: map_at_100
      value: 42.532
    - type: map_at_1000
      value: 42.661
    - type: map_at_3
      value: 37.483
    - type: map_at_5
      value: 39.652
    - type: mrr_at_1
      value: 36.338
    - type: mrr_at_10
      value: 46.763
    - type: mrr_at_100
      value: 47.393
    - type: mrr_at_1000
      value: 47.445
    - type: mrr_at_3
      value: 43.538
    - type: mrr_at_5
      value: 45.556000000000004
    - type: ndcg_at_1
      value: 36.338
    - type: ndcg_at_10
      value: 47.658
    - type: ndcg_at_100
      value: 52.824000000000005
    - type: ndcg_at_1000
      value: 54.913999999999994
    - type: ndcg_at_3
      value: 41.989
    - type: ndcg_at_5
      value: 44.944
    - type: precision_at_1
      value: 36.338
    - type: precision_at_10
      value: 9.156
    - type: precision_at_100
      value: 1.4789999999999999
    - type: precision_at_1000
      value: 0.196
    - type: precision_at_3
      value: 20.076
    - type: precision_at_5
      value: 14.85
    - type: recall_at_1
      value: 29.587000000000003
    - type: recall_at_10
      value: 60.746
    - type: recall_at_100
      value: 82.157
    - type: recall_at_1000
      value: 95.645
    - type: recall_at_3
      value: 44.821
    - type: recall_at_5
      value: 52.819
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackEnglishRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 30.239
    - type: map_at_10
      value: 39.989000000000004
    - type: map_at_100
      value: 41.196
    - type: map_at_1000
      value: 41.325
    - type: map_at_3
      value: 37.261
    - type: map_at_5
      value: 38.833
    - type: mrr_at_1
      value: 37.516
    - type: mrr_at_10
      value: 46.177
    - type: mrr_at_100
      value: 46.806
    - type: mrr_at_1000
      value: 46.849000000000004
    - type: mrr_at_3
      value: 44.002
    - type: mrr_at_5
      value: 45.34
    - type: ndcg_at_1
      value: 37.516
    - type: ndcg_at_10
      value: 45.586
    - type: ndcg_at_100
      value: 49.897000000000006
    - type: ndcg_at_1000
      value: 51.955
    - type: ndcg_at_3
      value: 41.684
    - type: ndcg_at_5
      value: 43.617
    - type: precision_at_1
      value: 37.516
    - type: precision_at_10
      value: 8.522
    - type: precision_at_100
      value: 1.374
    - type: precision_at_1000
      value: 0.184
    - type: precision_at_3
      value: 20.105999999999998
    - type: precision_at_5
      value: 14.152999999999999
    - type: recall_at_1
      value: 30.239
    - type: recall_at_10
      value: 55.03
    - type: recall_at_100
      value: 73.375
    - type: recall_at_1000
      value: 86.29599999999999
    - type: recall_at_3
      value: 43.269000000000005
    - type: recall_at_5
      value: 48.878
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackGamingRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 38.338
    - type: map_at_10
      value: 50.468999999999994
    - type: map_at_100
      value: 51.553000000000004
    - type: map_at_1000
      value: 51.608
    - type: map_at_3
      value: 47.107
    - type: map_at_5
      value: 49.101
    - type: mrr_at_1
      value: 44.201
    - type: mrr_at_10
      value: 54.057
    - type: mrr_at_100
      value: 54.764
    - type: mrr_at_1000
      value: 54.791000000000004
    - type: mrr_at_3
      value: 51.56699999999999
    - type: mrr_at_5
      value: 53.05
    - type: ndcg_at_1
      value: 44.201
    - type: ndcg_at_10
      value: 56.379000000000005
    - type: ndcg_at_100
      value: 60.645
    - type: ndcg_at_1000
      value: 61.73499999999999
    - type: ndcg_at_3
      value: 50.726000000000006
    - type: ndcg_at_5
      value: 53.58500000000001
    - type: precision_at_1
      value: 44.201
    - type: precision_at_10
      value: 9.141
    - type: precision_at_100
      value: 1.216
    - type: precision_at_1000
      value: 0.135
    - type: precision_at_3
      value: 22.654
    - type: precision_at_5
      value: 15.723999999999998
    - type: recall_at_1
      value: 38.338
    - type: recall_at_10
      value: 70.30499999999999
    - type: recall_at_100
      value: 88.77199999999999
    - type: recall_at_1000
      value: 96.49799999999999
    - type: recall_at_3
      value: 55.218
    - type: recall_at_5
      value: 62.104000000000006
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackGisRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 25.682
    - type: map_at_10
      value: 33.498
    - type: map_at_100
      value: 34.461000000000006
    - type: map_at_1000
      value: 34.544000000000004
    - type: map_at_3
      value: 30.503999999999998
    - type: map_at_5
      value: 32.216
    - type: mrr_at_1
      value: 27.683999999999997
    - type: mrr_at_10
      value: 35.467999999999996
    - type: mrr_at_100
      value: 36.32
    - type: mrr_at_1000
      value: 36.386
    - type: mrr_at_3
      value: 32.618
    - type: mrr_at_5
      value: 34.262
    - type: ndcg_at_1
      value: 27.683999999999997
    - type: ndcg_at_10
      value: 38.378
    - type: ndcg_at_100
      value: 43.288
    - type: ndcg_at_1000
      value: 45.413
    - type: ndcg_at_3
      value: 32.586
    - type: ndcg_at_5
      value: 35.499
    - type: precision_at_1
      value: 27.683999999999997
    - type: precision_at_10
      value: 5.864
    - type: precision_at_100
      value: 0.882
    - type: precision_at_1000
      value: 0.11
    - type: precision_at_3
      value: 13.446
    - type: precision_at_5
      value: 9.718
    - type: recall_at_1
      value: 25.682
    - type: recall_at_10
      value: 51.712
    - type: recall_at_100
      value: 74.446
    - type: recall_at_1000
      value: 90.472
    - type: recall_at_3
      value: 36.236000000000004
    - type: recall_at_5
      value: 43.234
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackMathematicaRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 16.073999999999998
    - type: map_at_10
      value: 24.352999999999998
    - type: map_at_100
      value: 25.438
    - type: map_at_1000
      value: 25.545
    - type: map_at_3
      value: 21.614
    - type: map_at_5
      value: 23.104
    - type: mrr_at_1
      value: 19.776
    - type: mrr_at_10
      value: 28.837000000000003
    - type: mrr_at_100
      value: 29.755
    - type: mrr_at_1000
      value: 29.817
    - type: mrr_at_3
      value: 26.201999999999998
    - type: mrr_at_5
      value: 27.714
    - type: ndcg_at_1
      value: 19.776
    - type: ndcg_at_10
      value: 29.701
    - type: ndcg_at_100
      value: 35.307
    - type: ndcg_at_1000
      value: 37.942
    - type: ndcg_at_3
      value: 24.764
    - type: ndcg_at_5
      value: 27.025
    - type: precision_at_1
      value: 19.776
    - type: precision_at_10
      value: 5.659
    - type: precision_at_100
      value: 0.971
    - type: precision_at_1000
      value: 0.133
    - type: precision_at_3
      value: 12.065
    - type: precision_at_5
      value: 8.905000000000001
    - type: recall_at_1
      value: 16.073999999999998
    - type: recall_at_10
      value: 41.647
    - type: recall_at_100
      value: 66.884
    - type: recall_at_1000
      value: 85.91499999999999
    - type: recall_at_3
      value: 27.916
    - type: recall_at_5
      value: 33.729
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackPhysicsRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 28.444999999999997
    - type: map_at_10
      value: 38.218999999999994
    - type: map_at_100
      value: 39.595
    - type: map_at_1000
      value: 39.709
    - type: map_at_3
      value: 35.586
    - type: map_at_5
      value: 36.895
    - type: mrr_at_1
      value: 34.841
    - type: mrr_at_10
      value: 44.106
    - type: mrr_at_100
      value: 44.98
    - type: mrr_at_1000
      value: 45.03
    - type: mrr_at_3
      value: 41.979
    - type: mrr_at_5
      value: 43.047999999999995
    - type: ndcg_at_1
      value: 34.841
    - type: ndcg_at_10
      value: 43.922
    - type: ndcg_at_100
      value: 49.504999999999995
    - type: ndcg_at_1000
      value: 51.675000000000004
    - type: ndcg_at_3
      value: 39.858
    - type: ndcg_at_5
      value: 41.408
    - type: precision_at_1
      value: 34.841
    - type: precision_at_10
      value: 7.872999999999999
    - type: precision_at_100
      value: 1.2449999999999999
    - type: precision_at_1000
      value: 0.161
    - type: precision_at_3
      value: 18.993
    - type: precision_at_5
      value: 13.032
    - type: recall_at_1
      value: 28.444999999999997
    - type: recall_at_10
      value: 54.984
    - type: recall_at_100
      value: 78.342
    - type: recall_at_1000
      value: 92.77
    - type: recall_at_3
      value: 42.842999999999996
    - type: recall_at_5
      value: 47.247
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackProgrammersRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 23.072
    - type: map_at_10
      value: 32.354
    - type: map_at_100
      value: 33.800000000000004
    - type: map_at_1000
      value: 33.908
    - type: map_at_3
      value: 29.232000000000003
    - type: map_at_5
      value: 31.049
    - type: mrr_at_1
      value: 29.110000000000003
    - type: mrr_at_10
      value: 38.03
    - type: mrr_at_100
      value: 39.032
    - type: mrr_at_1000
      value: 39.086999999999996
    - type: mrr_at_3
      value: 35.407
    - type: mrr_at_5
      value: 36.76
    - type: ndcg_at_1
      value: 29.110000000000003
    - type: ndcg_at_10
      value: 38.231
    - type: ndcg_at_100
      value: 44.425
    - type: ndcg_at_1000
      value: 46.771
    - type: ndcg_at_3
      value: 33.095
    - type: ndcg_at_5
      value: 35.459
    - type: precision_at_1
      value: 29.110000000000003
    - type: precision_at_10
      value: 7.215000000000001
    - type: precision_at_100
      value: 1.2109999999999999
    - type: precision_at_1000
      value: 0.157
    - type: precision_at_3
      value: 16.058
    - type: precision_at_5
      value: 11.644
    - type: recall_at_1
      value: 23.072
    - type: recall_at_10
      value: 50.285999999999994
    - type: recall_at_100
      value: 76.596
    - type: recall_at_1000
      value: 92.861
    - type: recall_at_3
      value: 35.702
    - type: recall_at_5
      value: 42.152
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 24.937916666666666
    - type: map_at_10
      value: 33.755250000000004
    - type: map_at_100
      value: 34.955999999999996
    - type: map_at_1000
      value: 35.070499999999996
    - type: map_at_3
      value: 30.98708333333333
    - type: map_at_5
      value: 32.51491666666666
    - type: mrr_at_1
      value: 29.48708333333333
    - type: mrr_at_10
      value: 37.92183333333334
    - type: mrr_at_100
      value: 38.76583333333333
    - type: mrr_at_1000
      value: 38.82466666666667
    - type: mrr_at_3
      value: 35.45125
    - type: mrr_at_5
      value: 36.827000000000005
    - type: ndcg_at_1
      value: 29.48708333333333
    - type: ndcg_at_10
      value: 39.05225
    - type: ndcg_at_100
      value: 44.25983333333334
    - type: ndcg_at_1000
      value: 46.568333333333335
    - type: ndcg_at_3
      value: 34.271583333333325
    - type: ndcg_at_5
      value: 36.483916666666666
    - type: precision_at_1
      value: 29.48708333333333
    - type: precision_at_10
      value: 6.865749999999999
    - type: precision_at_100
      value: 1.1195833333333332
    - type: precision_at_1000
      value: 0.15058333333333335
    - type: precision_at_3
      value: 15.742083333333333
    - type: precision_at_5
      value: 11.221916666666667
    - type: recall_at_1
      value: 24.937916666666666
    - type: recall_at_10
      value: 50.650416666666665
    - type: recall_at_100
      value: 73.55383333333334
    - type: recall_at_1000
      value: 89.61691666666667
    - type: recall_at_3
      value: 37.27808333333334
    - type: recall_at_5
      value: 42.99475
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackStatsRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 23.947
    - type: map_at_10
      value: 30.575000000000003
    - type: map_at_100
      value: 31.465
    - type: map_at_1000
      value: 31.558000000000003
    - type: map_at_3
      value: 28.814
    - type: map_at_5
      value: 29.738999999999997
    - type: mrr_at_1
      value: 26.994
    - type: mrr_at_10
      value: 33.415
    - type: mrr_at_100
      value: 34.18
    - type: mrr_at_1000
      value: 34.245
    - type: mrr_at_3
      value: 31.621
    - type: mrr_at_5
      value: 32.549
    - type: ndcg_at_1
      value: 26.994
    - type: ndcg_at_10
      value: 34.482
    - type: ndcg_at_100
      value: 38.915
    - type: ndcg_at_1000
      value: 41.355
    - type: ndcg_at_3
      value: 31.139
    - type: ndcg_at_5
      value: 32.589
    - type: precision_at_1
      value: 26.994
    - type: precision_at_10
      value: 5.322
    - type: precision_at_100
      value: 0.8160000000000001
    - type: precision_at_1000
      value: 0.11100000000000002
    - type: precision_at_3
      value: 13.344000000000001
    - type: precision_at_5
      value: 8.988
    - type: recall_at_1
      value: 23.947
    - type: recall_at_10
      value: 43.647999999999996
    - type: recall_at_100
      value: 63.851
    - type: recall_at_1000
      value: 82.0
    - type: recall_at_3
      value: 34.288000000000004
    - type: recall_at_5
      value: 38.117000000000004
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackTexRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 16.197
    - type: map_at_10
      value: 22.968
    - type: map_at_100
      value: 24.095
    - type: map_at_1000
      value: 24.217
    - type: map_at_3
      value: 20.771
    - type: map_at_5
      value: 21.995
    - type: mrr_at_1
      value: 19.511
    - type: mrr_at_10
      value: 26.55
    - type: mrr_at_100
      value: 27.500999999999998
    - type: mrr_at_1000
      value: 27.578999999999997
    - type: mrr_at_3
      value: 24.421
    - type: mrr_at_5
      value: 25.604
    - type: ndcg_at_1
      value: 19.511
    - type: ndcg_at_10
      value: 27.386
    - type: ndcg_at_100
      value: 32.828
    - type: ndcg_at_1000
      value: 35.739
    - type: ndcg_at_3
      value: 23.405
    - type: ndcg_at_5
      value: 25.255
    - type: precision_at_1
      value: 19.511
    - type: precision_at_10
      value: 5.017
    - type: precision_at_100
      value: 0.91
    - type: precision_at_1000
      value: 0.133
    - type: precision_at_3
      value: 11.023
    - type: precision_at_5
      value: 8.025
    - type: recall_at_1
      value: 16.197
    - type: recall_at_10
      value: 37.09
    - type: recall_at_100
      value: 61.778
    - type: recall_at_1000
      value: 82.56599999999999
    - type: recall_at_3
      value: 26.034000000000002
    - type: recall_at_5
      value: 30.762
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackUnixRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 25.41
    - type: map_at_10
      value: 33.655
    - type: map_at_100
      value: 34.892
    - type: map_at_1000
      value: 34.995
    - type: map_at_3
      value: 30.94
    - type: map_at_5
      value: 32.303
    - type: mrr_at_1
      value: 29.477999999999998
    - type: mrr_at_10
      value: 37.443
    - type: mrr_at_100
      value: 38.383
    - type: mrr_at_1000
      value: 38.440000000000005
    - type: mrr_at_3
      value: 34.949999999999996
    - type: mrr_at_5
      value: 36.228
    - type: ndcg_at_1
      value: 29.477999999999998
    - type: ndcg_at_10
      value: 38.769
    - type: ndcg_at_100
      value: 44.245000000000005
    - type: ndcg_at_1000
      value: 46.593
    - type: ndcg_at_3
      value: 33.623
    - type: ndcg_at_5
      value: 35.766
    - type: precision_at_1
      value: 29.477999999999998
    - type: precision_at_10
      value: 6.455
    - type: precision_at_100
      value: 1.032
    - type: precision_at_1000
      value: 0.135
    - type: precision_at_3
      value: 14.893999999999998
    - type: precision_at_5
      value: 10.485
    - type: recall_at_1
      value: 25.41
    - type: recall_at_10
      value: 50.669
    - type: recall_at_100
      value: 74.084
    - type: recall_at_1000
      value: 90.435
    - type: recall_at_3
      value: 36.679
    - type: recall_at_5
      value: 41.94
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackWebmastersRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 23.339
    - type: map_at_10
      value: 31.852000000000004
    - type: map_at_100
      value: 33.411
    - type: map_at_1000
      value: 33.62
    - type: map_at_3
      value: 28.929
    - type: map_at_5
      value: 30.542
    - type: mrr_at_1
      value: 28.063
    - type: mrr_at_10
      value: 36.301
    - type: mrr_at_100
      value: 37.288
    - type: mrr_at_1000
      value: 37.349
    - type: mrr_at_3
      value: 33.663
    - type: mrr_at_5
      value: 35.165
    - type: ndcg_at_1
      value: 28.063
    - type: ndcg_at_10
      value: 37.462
    - type: ndcg_at_100
      value: 43.620999999999995
    - type: ndcg_at_1000
      value: 46.211
    - type: ndcg_at_3
      value: 32.68
    - type: ndcg_at_5
      value: 34.981
    - type: precision_at_1
      value: 28.063
    - type: precision_at_10
      value: 7.1739999999999995
    - type: precision_at_100
      value: 1.486
    - type: precision_at_1000
      value: 0.23500000000000001
    - type: precision_at_3
      value: 15.217
    - type: precision_at_5
      value: 11.265
    - type: recall_at_1
      value: 23.339
    - type: recall_at_10
      value: 48.376999999999995
    - type: recall_at_100
      value: 76.053
    - type: recall_at_1000
      value: 92.455
    - type: recall_at_3
      value: 34.735
    - type: recall_at_5
      value: 40.71
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackWordpressRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 18.925
    - type: map_at_10
      value: 26.017000000000003
    - type: map_at_100
      value: 27.034000000000002
    - type: map_at_1000
      value: 27.156000000000002
    - type: map_at_3
      value: 23.604
    - type: map_at_5
      value: 24.75
    - type: mrr_at_1
      value: 20.333000000000002
    - type: mrr_at_10
      value: 27.915
    - type: mrr_at_100
      value: 28.788000000000004
    - type: mrr_at_1000
      value: 28.877999999999997
    - type: mrr_at_3
      value: 25.446999999999996
    - type: mrr_at_5
      value: 26.648
    - type: ndcg_at_1
      value: 20.333000000000002
    - type: ndcg_at_10
      value: 30.673000000000002
    - type: ndcg_at_100
      value: 35.618
    - type: ndcg_at_1000
      value: 38.517
    - type: ndcg_at_3
      value: 25.71
    - type: ndcg_at_5
      value: 27.679
    - type: precision_at_1
      value: 20.333000000000002
    - type: precision_at_10
      value: 4.9910000000000005
    - type: precision_at_100
      value: 0.8130000000000001
    - type: precision_at_1000
      value: 0.117
    - type: precision_at_3
      value: 11.029
    - type: precision_at_5
      value: 7.8740000000000006
    - type: recall_at_1
      value: 18.925
    - type: recall_at_10
      value: 43.311
    - type: recall_at_100
      value: 66.308
    - type: recall_at_1000
      value: 87.49
    - type: recall_at_3
      value: 29.596
    - type: recall_at_5
      value: 34.245
  - task:
      type: Retrieval
    dataset:
      type: climate-fever
      name: MTEB ClimateFEVER
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 13.714
    - type: map_at_10
      value: 23.194
    - type: map_at_100
      value: 24.976000000000003
    - type: map_at_1000
      value: 25.166
    - type: map_at_3
      value: 19.709
    - type: map_at_5
      value: 21.523999999999997
    - type: mrr_at_1
      value: 30.619000000000003
    - type: mrr_at_10
      value: 42.563
    - type: mrr_at_100
      value: 43.386
    - type: mrr_at_1000
      value: 43.423
    - type: mrr_at_3
      value: 39.555
    - type: mrr_at_5
      value: 41.268
    - type: ndcg_at_1
      value: 30.619000000000003
    - type: ndcg_at_10
      value: 31.836
    - type: ndcg_at_100
      value: 38.652
    - type: ndcg_at_1000
      value: 42.088
    - type: ndcg_at_3
      value: 26.733
    - type: ndcg_at_5
      value: 28.435
    - type: precision_at_1
      value: 30.619000000000003
    - type: precision_at_10
      value: 9.751999999999999
    - type: precision_at_100
      value: 1.71
    - type: precision_at_1000
      value: 0.23500000000000001
    - type: precision_at_3
      value: 19.935
    - type: precision_at_5
      value: 14.984
    - type: recall_at_1
      value: 13.714
    - type: recall_at_10
      value: 37.26
    - type: recall_at_100
      value: 60.546
    - type: recall_at_1000
      value: 79.899
    - type: recall_at_3
      value: 24.325
    - type: recall_at_5
      value: 29.725
  - task:
      type: Retrieval
    dataset:
      type: dbpedia-entity
      name: MTEB DBPedia
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 8.462
    - type: map_at_10
      value: 18.637
    - type: map_at_100
      value: 26.131999999999998
    - type: map_at_1000
      value: 27.607
    - type: map_at_3
      value: 13.333
    - type: map_at_5
      value: 15.654000000000002
    - type: mrr_at_1
      value: 66.25
    - type: mrr_at_10
      value: 74.32600000000001
    - type: mrr_at_100
      value: 74.60900000000001
    - type: mrr_at_1000
      value: 74.62
    - type: mrr_at_3
      value: 72.667
    - type: mrr_at_5
      value: 73.817
    - type: ndcg_at_1
      value: 53.87499999999999
    - type: ndcg_at_10
      value: 40.028999999999996
    - type: ndcg_at_100
      value: 44.199
    - type: ndcg_at_1000
      value: 51.629999999999995
    - type: ndcg_at_3
      value: 44.113
    - type: ndcg_at_5
      value: 41.731
    - type: precision_at_1
      value: 66.25
    - type: precision_at_10
      value: 31.900000000000002
    - type: precision_at_100
      value: 10.043000000000001
    - type: precision_at_1000
      value: 1.926
    - type: precision_at_3
      value: 47.417
    - type: precision_at_5
      value: 40.65
    - type: recall_at_1
      value: 8.462
    - type: recall_at_10
      value: 24.293
    - type: recall_at_100
      value: 50.146
    - type: recall_at_1000
      value: 74.034
    - type: recall_at_3
      value: 14.967
    - type: recall_at_5
      value: 18.682000000000002
  - task:
      type: Classification
    dataset:
      type: mteb/emotion
      name: MTEB EmotionClassification
      config: default
      split: test
      revision: 4f58c6b202a23cf9a4da393831edf4f9183cad37
    metrics:
    - type: accuracy
      value: 47.84499999999999
    - type: f1
      value: 42.48106691979349
  - task:
      type: Retrieval
    dataset:
      type: fever
      name: MTEB FEVER
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 74.034
    - type: map_at_10
      value: 82.76
    - type: map_at_100
      value: 82.968
    - type: map_at_1000
      value: 82.98299999999999
    - type: map_at_3
      value: 81.768
    - type: map_at_5
      value: 82.418
    - type: mrr_at_1
      value: 80.048
    - type: mrr_at_10
      value: 87.64999999999999
    - type: mrr_at_100
      value: 87.712
    - type: mrr_at_1000
      value: 87.713
    - type: mrr_at_3
      value: 87.01100000000001
    - type: mrr_at_5
      value: 87.466
    - type: ndcg_at_1
      value: 80.048
    - type: ndcg_at_10
      value: 86.643
    - type: ndcg_at_100
      value: 87.361
    - type: ndcg_at_1000
      value: 87.606
    - type: ndcg_at_3
      value: 85.137
    - type: ndcg_at_5
      value: 86.016
    - type: precision_at_1
      value: 80.048
    - type: precision_at_10
      value: 10.372
    - type: precision_at_100
      value: 1.093
    - type: precision_at_1000
      value: 0.11299999999999999
    - type: precision_at_3
      value: 32.638
    - type: precision_at_5
      value: 20.177
    - type: recall_at_1
      value: 74.034
    - type: recall_at_10
      value: 93.769
    - type: recall_at_100
      value: 96.569
    - type: recall_at_1000
      value: 98.039
    - type: recall_at_3
      value: 89.581
    - type: recall_at_5
      value: 91.906
  - task:
      type: Retrieval
    dataset:
      type: fiqa
      name: MTEB FiQA2018
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 20.5
    - type: map_at_10
      value: 32.857
    - type: map_at_100
      value: 34.589
    - type: map_at_1000
      value: 34.778
    - type: map_at_3
      value: 29.160999999999998
    - type: map_at_5
      value: 31.033
    - type: mrr_at_1
      value: 40.123
    - type: mrr_at_10
      value: 48.776
    - type: mrr_at_100
      value: 49.495
    - type: mrr_at_1000
      value: 49.539
    - type: mrr_at_3
      value: 46.605000000000004
    - type: mrr_at_5
      value: 47.654
    - type: ndcg_at_1
      value: 40.123
    - type: ndcg_at_10
      value: 40.343
    - type: ndcg_at_100
      value: 46.56
    - type: ndcg_at_1000
      value: 49.777
    - type: ndcg_at_3
      value: 37.322
    - type: ndcg_at_5
      value: 37.791000000000004
    - type: precision_at_1
      value: 40.123
    - type: precision_at_10
      value: 11.08
    - type: precision_at_100
      value: 1.752
    - type: precision_at_1000
      value: 0.232
    - type: precision_at_3
      value: 24.897
    - type: precision_at_5
      value: 17.809
    - type: recall_at_1
      value: 20.5
    - type: recall_at_10
      value: 46.388
    - type: recall_at_100
      value: 69.552
    - type: recall_at_1000
      value: 89.011
    - type: recall_at_3
      value: 33.617999999999995
    - type: recall_at_5
      value: 38.211
  - task:
      type: Retrieval
    dataset:
      type: hotpotqa
      name: MTEB HotpotQA
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 39.135999999999996
    - type: map_at_10
      value: 61.673
    - type: map_at_100
      value: 62.562
    - type: map_at_1000
      value: 62.62
    - type: map_at_3
      value: 58.467999999999996
    - type: map_at_5
      value: 60.463
    - type: mrr_at_1
      value: 78.271
    - type: mrr_at_10
      value: 84.119
    - type: mrr_at_100
      value: 84.29299999999999
    - type: mrr_at_1000
      value: 84.299
    - type: mrr_at_3
      value: 83.18900000000001
    - type: mrr_at_5
      value: 83.786
    - type: ndcg_at_1
      value: 78.271
    - type: ndcg_at_10
      value: 69.935
    - type: ndcg_at_100
      value: 73.01299999999999
    - type: ndcg_at_1000
      value: 74.126
    - type: ndcg_at_3
      value: 65.388
    - type: ndcg_at_5
      value: 67.906
    - type: precision_at_1
      value: 78.271
    - type: precision_at_10
      value: 14.562
    - type: precision_at_100
      value: 1.6969999999999998
    - type: precision_at_1000
      value: 0.184
    - type: precision_at_3
      value: 41.841
    - type: precision_at_5
      value: 27.087
    - type: recall_at_1
      value: 39.135999999999996
    - type: recall_at_10
      value: 72.809
    - type: recall_at_100
      value: 84.86200000000001
    - type: recall_at_1000
      value: 92.208
    - type: recall_at_3
      value: 62.76199999999999
    - type: recall_at_5
      value: 67.718
  - task:
      type: Classification
    dataset:
      type: mteb/imdb
      name: MTEB ImdbClassification
      config: default
      split: test
      revision: 3d86128a09e091d6018b6d26cad27f2739fc2db7
    metrics:
    - type: accuracy
      value: 90.60600000000001
    - type: ap
      value: 86.6579587804335
    - type: f1
      value: 90.5938853929307
  - task:
      type: Retrieval
    dataset:
      type: msmarco
      name: MTEB MSMARCO
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 21.852
    - type: map_at_10
      value: 33.982
    - type: map_at_100
      value: 35.116
    - type: map_at_1000
      value: 35.167
    - type: map_at_3
      value: 30.134
    - type: map_at_5
      value: 32.340999999999994
    - type: mrr_at_1
      value: 22.479
    - type: mrr_at_10
      value: 34.594
    - type: mrr_at_100
      value: 35.672
    - type: mrr_at_1000
      value: 35.716
    - type: mrr_at_3
      value: 30.84
    - type: mrr_at_5
      value: 32.998
    - type: ndcg_at_1
      value: 22.493
    - type: ndcg_at_10
      value: 40.833000000000006
    - type: ndcg_at_100
      value: 46.357
    - type: ndcg_at_1000
      value: 47.637
    - type: ndcg_at_3
      value: 32.995999999999995
    - type: ndcg_at_5
      value: 36.919000000000004
    - type: precision_at_1
      value: 22.493
    - type: precision_at_10
      value: 6.465999999999999
    - type: precision_at_100
      value: 0.9249999999999999
    - type: precision_at_1000
      value: 0.104
    - type: precision_at_3
      value: 14.030999999999999
    - type: precision_at_5
      value: 10.413
    - type: recall_at_1
      value: 21.852
    - type: recall_at_10
      value: 61.934999999999995
    - type: recall_at_100
      value: 87.611
    - type: recall_at_1000
      value: 97.441
    - type: recall_at_3
      value: 40.583999999999996
    - type: recall_at_5
      value: 49.992999999999995
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_domain
      name: MTEB MTOPDomainClassification (en)
      config: en
      split: test
      revision: d80d48c1eb48d3562165c59d59d0034df9fff0bf
    metrics:
    - type: accuracy
      value: 93.36069311445507
    - type: f1
      value: 93.16456330371453
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_intent
      name: MTEB MTOPIntentClassification (en)
      config: en
      split: test
      revision: ae001d0e6b1228650b7bd1c2c65fb50ad11a8aba
    metrics:
    - type: accuracy
      value: 74.74692202462381
    - type: f1
      value: 58.17903579421599
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (en)
      config: en
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 74.80833893745796
    - type: f1
      value: 72.70786592684664
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (en)
      config: en
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 78.69872225958305
    - type: f1
      value: 78.61626934504731
  - task:
      type: Clustering
    dataset:
      type: mteb/medrxiv-clustering-p2p
      name: MTEB MedrxivClusteringP2P
      config: default
      split: test
      revision: e7a26af6f3ae46b30dde8737f02c07b1505bcc73
    metrics:
    - type: v_measure
      value: 33.058658628717694
  - task:
      type: Clustering
    dataset:
      type: mteb/medrxiv-clustering-s2s
      name: MTEB MedrxivClusteringS2S
      config: default
      split: test
      revision: 35191c8c0dca72d8ff3efcd72aa802307d469663
    metrics:
    - type: v_measure
      value: 30.85561739360599
  - task:
      type: Reranking
    dataset:
      type: mteb/mind_small
      name: MTEB MindSmallReranking
      config: default
      split: test
      revision: 3bdac13927fdc888b903db93b2ffdbd90b295a69
    metrics:
    - type: map
      value: 31.290259910144385
    - type: mrr
      value: 32.44223046102856
  - task:
      type: Retrieval
    dataset:
      type: nfcorpus
      name: MTEB NFCorpus
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 5.288
    - type: map_at_10
      value: 12.267999999999999
    - type: map_at_100
      value: 15.557000000000002
    - type: map_at_1000
      value: 16.98
    - type: map_at_3
      value: 8.866
    - type: map_at_5
      value: 10.418
    - type: mrr_at_1
      value: 43.653
    - type: mrr_at_10
      value: 52.681
    - type: mrr_at_100
      value: 53.315999999999995
    - type: mrr_at_1000
      value: 53.357
    - type: mrr_at_3
      value: 51.393
    - type: mrr_at_5
      value: 51.903999999999996
    - type: ndcg_at_1
      value: 42.415000000000006
    - type: ndcg_at_10
      value: 34.305
    - type: ndcg_at_100
      value: 30.825999999999997
    - type: ndcg_at_1000
      value: 39.393
    - type: ndcg_at_3
      value: 39.931
    - type: ndcg_at_5
      value: 37.519999999999996
    - type: precision_at_1
      value: 43.653
    - type: precision_at_10
      value: 25.728
    - type: precision_at_100
      value: 7.932
    - type: precision_at_1000
      value: 2.07
    - type: precision_at_3
      value: 38.184000000000005
    - type: precision_at_5
      value: 32.879000000000005
    - type: recall_at_1
      value: 5.288
    - type: recall_at_10
      value: 16.195
    - type: recall_at_100
      value: 31.135
    - type: recall_at_1000
      value: 61.531000000000006
    - type: recall_at_3
      value: 10.313
    - type: recall_at_5
      value: 12.754999999999999
  - task:
      type: Retrieval
    dataset:
      type: nq
      name: MTEB NQ
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 28.216
    - type: map_at_10
      value: 42.588
    - type: map_at_100
      value: 43.702999999999996
    - type: map_at_1000
      value: 43.739
    - type: map_at_3
      value: 38.177
    - type: map_at_5
      value: 40.754000000000005
    - type: mrr_at_1
      value: 31.866
    - type: mrr_at_10
      value: 45.189
    - type: mrr_at_100
      value: 46.056000000000004
    - type: mrr_at_1000
      value: 46.081
    - type: mrr_at_3
      value: 41.526999999999994
    - type: mrr_at_5
      value: 43.704
    - type: ndcg_at_1
      value: 31.837
    - type: ndcg_at_10
      value: 50.178
    - type: ndcg_at_100
      value: 54.98800000000001
    - type: ndcg_at_1000
      value: 55.812
    - type: ndcg_at_3
      value: 41.853
    - type: ndcg_at_5
      value: 46.153
    - type: precision_at_1
      value: 31.837
    - type: precision_at_10
      value: 8.43
    - type: precision_at_100
      value: 1.1119999999999999
    - type: precision_at_1000
      value: 0.11900000000000001
    - type: precision_at_3
      value: 19.023
    - type: precision_at_5
      value: 13.911000000000001
    - type: recall_at_1
      value: 28.216
    - type: recall_at_10
      value: 70.8
    - type: recall_at_100
      value: 91.857
    - type: recall_at_1000
      value: 97.941
    - type: recall_at_3
      value: 49.196
    - type: recall_at_5
      value: 59.072
  - task:
      type: Retrieval
    dataset:
      type: quora
      name: MTEB QuoraRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 71.22800000000001
    - type: map_at_10
      value: 85.115
    - type: map_at_100
      value: 85.72
    - type: map_at_1000
      value: 85.737
    - type: map_at_3
      value: 82.149
    - type: map_at_5
      value: 84.029
    - type: mrr_at_1
      value: 81.96
    - type: mrr_at_10
      value: 88.00200000000001
    - type: mrr_at_100
      value: 88.088
    - type: mrr_at_1000
      value: 88.089
    - type: mrr_at_3
      value: 87.055
    - type: mrr_at_5
      value: 87.715
    - type: ndcg_at_1
      value: 82.01
    - type: ndcg_at_10
      value: 88.78
    - type: ndcg_at_100
      value: 89.91
    - type: ndcg_at_1000
      value: 90.013
    - type: ndcg_at_3
      value: 85.957
    - type: ndcg_at_5
      value: 87.56
    - type: precision_at_1
      value: 82.01
    - type: precision_at_10
      value: 13.462
    - type: precision_at_100
      value: 1.528
    - type: precision_at_1000
      value: 0.157
    - type: precision_at_3
      value: 37.553
    - type: precision_at_5
      value: 24.732000000000003
    - type: recall_at_1
      value: 71.22800000000001
    - type: recall_at_10
      value: 95.69
    - type: recall_at_100
      value: 99.531
    - type: recall_at_1000
      value: 99.98
    - type: recall_at_3
      value: 87.632
    - type: recall_at_5
      value: 92.117
  - task:
      type: Clustering
    dataset:
      type: mteb/reddit-clustering
      name: MTEB RedditClustering
      config: default
      split: test
      revision: 24640382cdbf8abc73003fb0fa6d111a705499eb
    metrics:
    - type: v_measure
      value: 52.31768034366916
  - task:
      type: Clustering
    dataset:
      type: mteb/reddit-clustering-p2p
      name: MTEB RedditClusteringP2P
      config: default
      split: test
      revision: 282350215ef01743dc01b456c7f5241fa8937f16
    metrics:
    - type: v_measure
      value: 60.640266772723606
  - task:
      type: Retrieval
    dataset:
      type: scidocs
      name: MTEB SCIDOCS
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 4.7780000000000005
    - type: map_at_10
      value: 12.299
    - type: map_at_100
      value: 14.363000000000001
    - type: map_at_1000
      value: 14.71
    - type: map_at_3
      value: 8.738999999999999
    - type: map_at_5
      value: 10.397
    - type: mrr_at_1
      value: 23.599999999999998
    - type: mrr_at_10
      value: 34.845
    - type: mrr_at_100
      value: 35.916
    - type: mrr_at_1000
      value: 35.973
    - type: mrr_at_3
      value: 31.7
    - type: mrr_at_5
      value: 33.535
    - type: ndcg_at_1
      value: 23.599999999999998
    - type: ndcg_at_10
      value: 20.522000000000002
    - type: ndcg_at_100
      value: 28.737000000000002
    - type: ndcg_at_1000
      value: 34.596
    - type: ndcg_at_3
      value: 19.542
    - type: ndcg_at_5
      value: 16.958000000000002
    - type: precision_at_1
      value: 23.599999999999998
    - type: precision_at_10
      value: 10.67
    - type: precision_at_100
      value: 2.259
    - type: precision_at_1000
      value: 0.367
    - type: precision_at_3
      value: 18.333
    - type: precision_at_5
      value: 14.879999999999999
    - type: recall_at_1
      value: 4.7780000000000005
    - type: recall_at_10
      value: 21.617
    - type: recall_at_100
      value: 45.905
    - type: recall_at_1000
      value: 74.42
    - type: recall_at_3
      value: 11.148
    - type: recall_at_5
      value: 15.082999999999998
  - task:
      type: STS
    dataset:
      type: mteb/sickr-sts
      name: MTEB SICK-R
      config: default
      split: test
      revision: a6ea5a8cab320b040a23452cc28066d9beae2cee
    metrics:
    - type: cos_sim_pearson
      value: 83.22372750297885
    - type: cos_sim_spearman
      value: 79.40972617119405
    - type: euclidean_pearson
      value: 80.6101072020434
    - type: euclidean_spearman
      value: 79.53844217225202
    - type: manhattan_pearson
      value: 80.57265975286111
    - type: manhattan_spearman
      value: 79.46335611792958
  - task:
      type: STS
    dataset:
      type: mteb/sts12-sts
      name: MTEB STS12
      config: default
      split: test
      revision: a0d554a64d88156834ff5ae9920b964011b16384
    metrics:
    - type: cos_sim_pearson
      value: 85.43713315520749
    - type: cos_sim_spearman
      value: 77.44128693329532
    - type: euclidean_pearson
      value: 81.63869928101123
    - type: euclidean_spearman
      value: 77.29512977961515
    - type: manhattan_pearson
      value: 81.63704185566183
    - type: manhattan_spearman
      value: 77.29909412738657
  - task:
      type: STS
    dataset:
      type: mteb/sts13-sts
      name: MTEB STS13
      config: default
      split: test
      revision: 7e90230a92c190f1bf69ae9002b8cea547a64cca
    metrics:
    - type: cos_sim_pearson
      value: 81.59451537860527
    - type: cos_sim_spearman
      value: 82.97994638856723
    - type: euclidean_pearson
      value: 82.89478688288412
    - type: euclidean_spearman
      value: 83.58740751053104
    - type: manhattan_pearson
      value: 82.69140840941608
    - type: manhattan_spearman
      value: 83.33665956040555
  - task:
      type: STS
    dataset:
      type: mteb/sts14-sts
      name: MTEB STS14
      config: default
      split: test
      revision: 6031580fec1f6af667f0bd2da0a551cf4f0b2375
    metrics:
    - type: cos_sim_pearson
      value: 82.00756527711764
    - type: cos_sim_spearman
      value: 81.83560996841379
    - type: euclidean_pearson
      value: 82.07684151976518
    - type: euclidean_spearman
      value: 82.00913052060511
    - type: manhattan_pearson
      value: 82.05690778488794
    - type: manhattan_spearman
      value: 82.02260252019525
  - task:
      type: STS
    dataset:
      type: mteb/sts15-sts
      name: MTEB STS15
      config: default
      split: test
      revision: ae752c7c21bf194d8b67fd573edf7ae58183cbe3
    metrics:
    - type: cos_sim_pearson
      value: 86.13710262895447
    - type: cos_sim_spearman
      value: 87.26412811156248
    - type: euclidean_pearson
      value: 86.94151453230228
    - type: euclidean_spearman
      value: 87.5363796699571
    - type: manhattan_pearson
      value: 86.86989424083748
    - type: manhattan_spearman
      value: 87.47315940781353
  - task:
      type: STS
    dataset:
      type: mteb/sts16-sts
      name: MTEB STS16
      config: default
      split: test
      revision: 4d8694f8f0e0100860b497b999b3dbed754a0513
    metrics:
    - type: cos_sim_pearson
      value: 83.0230597603627
    - type: cos_sim_spearman
      value: 84.93344499318864
    - type: euclidean_pearson
      value: 84.23754743431141
    - type: euclidean_spearman
      value: 85.09707376597099
    - type: manhattan_pearson
      value: 84.04325160987763
    - type: manhattan_spearman
      value: 84.89353071339909
  - task:
      type: STS
    dataset:
      type: mteb/sts17-crosslingual-sts
      name: MTEB STS17 (en-en)
      config: en-en
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_pearson
      value: 86.75620824563921
    - type: cos_sim_spearman
      value: 87.15065513706398
    - type: euclidean_pearson
      value: 88.26281533633521
    - type: euclidean_spearman
      value: 87.51963738643983
    - type: manhattan_pearson
      value: 88.25599267618065
    - type: manhattan_spearman
      value: 87.58048736047483
  - task:
      type: STS
    dataset:
      type: mteb/sts22-crosslingual-sts
      name: MTEB STS22 (en)
      config: en
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 64.74645319195137
    - type: cos_sim_spearman
      value: 65.29996325037214
    - type: euclidean_pearson
      value: 67.04297794086443
    - type: euclidean_spearman
      value: 65.43841726694343
    - type: manhattan_pearson
      value: 67.39459955690904
    - type: manhattan_spearman
      value: 65.92864704413651
  - task:
      type: STS
    dataset:
      type: mteb/stsbenchmark-sts
      name: MTEB STSBenchmark
      config: default
      split: test
      revision: b0fddb56ed78048fa8b90373c8a3cfc37b684831
    metrics:
    - type: cos_sim_pearson
      value: 84.31291020270801
    - type: cos_sim_spearman
      value: 85.86473738688068
    - type: euclidean_pearson
      value: 85.65537275064152
    - type: euclidean_spearman
      value: 86.13087454209642
    - type: manhattan_pearson
      value: 85.43946955047609
    - type: manhattan_spearman
      value: 85.91568175344916
  - task:
      type: Reranking
    dataset:
      type: mteb/scidocs-reranking
      name: MTEB SciDocsRR
      config: default
      split: test
      revision: d3c5e1fc0b855ab6097bf1cda04dd73947d7caab
    metrics:
    - type: map
      value: 85.93798118350695
    - type: mrr
      value: 95.93536274908824
  - task:
      type: Retrieval
    dataset:
      type: scifact
      name: MTEB SciFact
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 57.594
    - type: map_at_10
      value: 66.81899999999999
    - type: map_at_100
      value: 67.368
    - type: map_at_1000
      value: 67.4
    - type: map_at_3
      value: 64.061
    - type: map_at_5
      value: 65.47
    - type: mrr_at_1
      value: 60.667
    - type: mrr_at_10
      value: 68.219
    - type: mrr_at_100
      value: 68.655
    - type: mrr_at_1000
      value: 68.684
    - type: mrr_at_3
      value: 66.22200000000001
    - type: mrr_at_5
      value: 67.289
    - type: ndcg_at_1
      value: 60.667
    - type: ndcg_at_10
      value: 71.275
    - type: ndcg_at_100
      value: 73.642
    - type: ndcg_at_1000
      value: 74.373
    - type: ndcg_at_3
      value: 66.521
    - type: ndcg_at_5
      value: 68.581
    - type: precision_at_1
      value: 60.667
    - type: precision_at_10
      value: 9.433
    - type: precision_at_100
      value: 1.0699999999999998
    - type: precision_at_1000
      value: 0.11299999999999999
    - type: precision_at_3
      value: 25.556
    - type: precision_at_5
      value: 16.8
    - type: recall_at_1
      value: 57.594
    - type: recall_at_10
      value: 83.622
    - type: recall_at_100
      value: 94.167
    - type: recall_at_1000
      value: 99.667
    - type: recall_at_3
      value: 70.64399999999999
    - type: recall_at_5
      value: 75.983
  - task:
      type: PairClassification
    dataset:
      type: mteb/sprintduplicatequestions-pairclassification
      name: MTEB SprintDuplicateQuestions
      config: default
      split: test
      revision: d66bd1f72af766a5cc4b0ca5e00c162f89e8cc46
    metrics:
    - type: cos_sim_accuracy
      value: 99.85841584158416
    - type: cos_sim_ap
      value: 96.66996142314342
    - type: cos_sim_f1
      value: 92.83208020050125
    - type: cos_sim_precision
      value: 93.06532663316584
    - type: cos_sim_recall
      value: 92.60000000000001
    - type: dot_accuracy
      value: 99.85841584158416
    - type: dot_ap
      value: 96.6775307676576
    - type: dot_f1
      value: 92.69289729177312
    - type: dot_precision
      value: 94.77533960292581
    - type: dot_recall
      value: 90.7
    - type: euclidean_accuracy
      value: 99.86138613861387
    - type: euclidean_ap
      value: 96.6338454403108
    - type: euclidean_f1
      value: 92.92214357937311
    - type: euclidean_precision
      value: 93.96728016359918
    - type: euclidean_recall
      value: 91.9
    - type: manhattan_accuracy
      value: 99.86237623762376
    - type: manhattan_ap
      value: 96.60370449645053
    - type: manhattan_f1
      value: 92.91177970423253
    - type: manhattan_precision
      value: 94.7970863683663
    - type: manhattan_recall
      value: 91.10000000000001
    - type: max_accuracy
      value: 99.86237623762376
    - type: max_ap
      value: 96.6775307676576
    - type: max_f1
      value: 92.92214357937311
  - task:
      type: Clustering
    dataset:
      type: mteb/stackexchange-clustering
      name: MTEB StackExchangeClustering
      config: default
      split: test
      revision: 6cbc1f7b2bc0622f2e39d2c77fa502909748c259
    metrics:
    - type: v_measure
      value: 60.77977058695198
  - task:
      type: Clustering
    dataset:
      type: mteb/stackexchange-clustering-p2p
      name: MTEB StackExchangeClusteringP2P
      config: default
      split: test
      revision: 815ca46b2622cec33ccafc3735d572c266efdb44
    metrics:
    - type: v_measure
      value: 35.2725272535638
  - task:
      type: Reranking
    dataset:
      type: mteb/stackoverflowdupquestions-reranking
      name: MTEB StackOverflowDupQuestions
      config: default
      split: test
      revision: e185fbe320c72810689fc5848eb6114e1ef5ec69
    metrics:
    - type: map
      value: 53.64052466362125
    - type: mrr
      value: 54.533067014684654
  - task:
      type: Summarization
    dataset:
      type: mteb/summeval
      name: MTEB SummEval
      config: default
      split: test
      revision: cda12ad7615edc362dbf25a00fdd61d3b1eaf93c
    metrics:
    - type: cos_sim_pearson
      value: 30.677624219206578
    - type: cos_sim_spearman
      value: 30.121368518123447
    - type: dot_pearson
      value: 30.69870088041608
    - type: dot_spearman
      value: 29.61284927093751
  - task:
      type: Retrieval
    dataset:
      type: trec-covid
      name: MTEB TRECCOVID
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 0.22
    - type: map_at_10
      value: 1.855
    - type: map_at_100
      value: 9.885
    - type: map_at_1000
      value: 23.416999999999998
    - type: map_at_3
      value: 0.637
    - type: map_at_5
      value: 1.024
    - type: mrr_at_1
      value: 88.0
    - type: mrr_at_10
      value: 93.067
    - type: mrr_at_100
      value: 93.067
    - type: mrr_at_1000
      value: 93.067
    - type: mrr_at_3
      value: 92.667
    - type: mrr_at_5
      value: 93.067
    - type: ndcg_at_1
      value: 82.0
    - type: ndcg_at_10
      value: 75.899
    - type: ndcg_at_100
      value: 55.115
    - type: ndcg_at_1000
      value: 48.368
    - type: ndcg_at_3
      value: 79.704
    - type: ndcg_at_5
      value: 78.39699999999999
    - type: precision_at_1
      value: 88.0
    - type: precision_at_10
      value: 79.60000000000001
    - type: precision_at_100
      value: 56.06
    - type: precision_at_1000
      value: 21.206
    - type: precision_at_3
      value: 84.667
    - type: precision_at_5
      value: 83.2
    - type: recall_at_1
      value: 0.22
    - type: recall_at_10
      value: 2.078
    - type: recall_at_100
      value: 13.297
    - type: recall_at_1000
      value: 44.979
    - type: recall_at_3
      value: 0.6689999999999999
    - type: recall_at_5
      value: 1.106
  - task:
      type: Retrieval
    dataset:
      type: webis-touche2020
      name: MTEB Touche2020
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 2.258
    - type: map_at_10
      value: 10.439
    - type: map_at_100
      value: 16.89
    - type: map_at_1000
      value: 18.407999999999998
    - type: map_at_3
      value: 5.668
    - type: map_at_5
      value: 7.718
    - type: mrr_at_1
      value: 32.653
    - type: mrr_at_10
      value: 51.159
    - type: mrr_at_100
      value: 51.714000000000006
    - type: mrr_at_1000
      value: 51.714000000000006
    - type: mrr_at_3
      value: 47.959
    - type: mrr_at_5
      value: 50.407999999999994
    - type: ndcg_at_1
      value: 29.592000000000002
    - type: ndcg_at_10
      value: 26.037
    - type: ndcg_at_100
      value: 37.924
    - type: ndcg_at_1000
      value: 49.126999999999995
    - type: ndcg_at_3
      value: 30.631999999999998
    - type: ndcg_at_5
      value: 28.571
    - type: precision_at_1
      value: 32.653
    - type: precision_at_10
      value: 22.857
    - type: precision_at_100
      value: 7.754999999999999
    - type: precision_at_1000
      value: 1.529
    - type: precision_at_3
      value: 34.014
    - type: precision_at_5
      value: 29.796
    - type: recall_at_1
      value: 2.258
    - type: recall_at_10
      value: 16.554
    - type: recall_at_100
      value: 48.439
    - type: recall_at_1000
      value: 82.80499999999999
    - type: recall_at_3
      value: 7.283
    - type: recall_at_5
      value: 10.732
  - task:
      type: Classification
    dataset:
      type: mteb/toxic_conversations_50k
      name: MTEB ToxicConversationsClassification
      config: default
      split: test
      revision: d7c0de2777da35d6aae2200a62c6e0e5af397c4c
    metrics:
    - type: accuracy
      value: 69.8858
    - type: ap
      value: 13.835684144362109
    - type: f1
      value: 53.803351693244586
  - task:
      type: Classification
    dataset:
      type: mteb/tweet_sentiment_extraction
      name: MTEB TweetSentimentExtractionClassification
      config: default
      split: test
      revision: d604517c81ca91fe16a244d1248fc021f9ecee7a
    metrics:
    - type: accuracy
      value: 60.50650820599886
    - type: f1
      value: 60.84357825979259
  - task:
      type: Clustering
    dataset:
      type: mteb/twentynewsgroups-clustering
      name: MTEB TwentyNewsgroupsClustering
      config: default
      split: test
      revision: 6125ec4e24fa026cec8a478383ee943acfbd5449
    metrics:
    - type: v_measure
      value: 48.52131044852134
  - task:
      type: PairClassification
    dataset:
      type: mteb/twittersemeval2015-pairclassification
      name: MTEB TwitterSemEval2015
      config: default
      split: test
      revision: 70970daeab8776df92f5ea462b6173c0b46fd2d1
    metrics:
    - type: cos_sim_accuracy
      value: 85.59337187816654
    - type: cos_sim_ap
      value: 73.23925826533437
    - type: cos_sim_f1
      value: 67.34693877551021
    - type: cos_sim_precision
      value: 62.40432237730752
    - type: cos_sim_recall
      value: 73.13984168865434
    - type: dot_accuracy
      value: 85.31322644096085
    - type: dot_ap
      value: 72.30723963807422
    - type: dot_f1
      value: 66.47051612112296
    - type: dot_precision
      value: 62.0792305930845
    - type: dot_recall
      value: 71.53034300791556
    - type: euclidean_accuracy
      value: 85.61125350181797
    - type: euclidean_ap
      value: 73.32843720487845
    - type: euclidean_f1
      value: 67.36549633745895
    - type: euclidean_precision
      value: 64.60755813953489
    - type: euclidean_recall
      value: 70.36939313984169
    - type: manhattan_accuracy
      value: 85.63509566668654
    - type: manhattan_ap
      value: 73.16658488311325
    - type: manhattan_f1
      value: 67.20597386434349
    - type: manhattan_precision
      value: 63.60424028268551
    - type: manhattan_recall
      value: 71.2401055408971
    - type: max_accuracy
      value: 85.63509566668654
    - type: max_ap
      value: 73.32843720487845
    - type: max_f1
      value: 67.36549633745895
  - task:
      type: PairClassification
    dataset:
      type: mteb/twitterurlcorpus-pairclassification
      name: MTEB TwitterURLCorpus
      config: default
      split: test
      revision: 8b6510b0b1fa4e4c4f879467980e9be563ec1cdf
    metrics:
    - type: cos_sim_accuracy
      value: 88.33779640625606
    - type: cos_sim_ap
      value: 84.83868375898157
    - type: cos_sim_f1
      value: 77.16506154017773
    - type: cos_sim_precision
      value: 74.62064005753327
    - type: cos_sim_recall
      value: 79.88912842623961
    - type: dot_accuracy
      value: 88.02732176815307
    - type: dot_ap
      value: 83.95089283763002
    - type: dot_f1
      value: 76.29635101196631
    - type: dot_precision
      value: 73.31771720613288
    - type: dot_recall
      value: 79.52725592854944
    - type: euclidean_accuracy
      value: 88.44452206310397
    - type: euclidean_ap
      value: 84.98384576824827
    - type: euclidean_f1
      value: 77.29311047696697
    - type: euclidean_precision
      value: 74.51232583065381
    - type: euclidean_recall
      value: 80.28949799815214
    - type: manhattan_accuracy
      value: 88.47362906042613
    - type: manhattan_ap
      value: 84.91421462218432
    - type: manhattan_f1
      value: 77.05107637204792
    - type: manhattan_precision
      value: 74.74484256243214
    - type: manhattan_recall
      value: 79.50415768401602
    - type: max_accuracy
      value: 88.47362906042613
    - type: max_ap
      value: 84.98384576824827
    - type: max_f1
      value: 77.29311047696697
license: mit
language:
- en
---


<h1 align="center">FlagEmbedding</h1>


<h4 align="center">
    <p>
        <a href=#model-list>Model List</a> | 
        <a href=#frequently-asked-questions>FAQ</a> |
        <a href=#usage>Usage</a>  |
        <a href="#evaluation">Evaluation</a> |
        <a href="#train">Train</a> |
        <a href="#contact">Contact</a> |
        <a href="#citation">Citation</a> |
        <a href="#license">License</a> 
    <p>
</h4>

More details please refer to our Github: [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding).

If you are looking for a model that supports more languages, longer texts, and other retrieval methods, you can try using [bge-m3](https://huggingface.co/BAAI/bge-m3).


[English](README.md) | [中文](https://github.com/FlagOpen/FlagEmbedding/blob/master/README_zh.md)

FlagEmbedding focuses on retrieval-augmented LLMs, consisting of the following projects currently:

- **Long-Context LLM**: [Activation Beacon](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/activation_beacon)
- **Fine-tuning of LM** : [LM-Cocktail](https://github.com/FlagOpen/FlagEmbedding/tree/master/LM_Cocktail)
- **Dense Retrieval**: [BGE-M3](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/BGE_M3), [LLM Embedder](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder), [BGE Embedding](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding)
- **Reranker Model**: [BGE Reranker](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker)
- **Benchmark**: [C-MTEB](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB)

## News 
- 1/30/2024: Release **BGE-M3**, a new member to BGE model series! M3 stands for **M**ulti-linguality (100+ languages), **M**ulti-granularities (input length up to 8192), **M**ulti-Functionality (unification of dense, lexical, multi-vec/colbert retrieval). 
It is the first embedding model which supports all three retrieval methods, achieving new SOTA on multi-lingual (MIRACL) and cross-lingual (MKQA) benchmarks.
[Technical Report](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/BGE_M3/BGE_M3.pdf) and [Code](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/BGE_M3). :fire:
- 1/9/2024: Release [Activation-Beacon](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/activation_beacon), an effective, efficient, compatible, and low-cost (training) method to extend the context length of LLM. [Technical Report](https://arxiv.org/abs/2401.03462) :fire:
- 12/24/2023: Release **LLaRA**, a LLaMA-7B based dense retriever, leading to state-of-the-art performances on MS MARCO and BEIR. Model and code will be open-sourced. Please stay tuned. [Technical Report](https://arxiv.org/abs/2312.15503) :fire:
- 11/23/2023: Release [LM-Cocktail](https://github.com/FlagOpen/FlagEmbedding/tree/master/LM_Cocktail), a method to maintain general capabilities during fine-tuning by merging multiple language models. [Technical Report](https://arxiv.org/abs/2311.13534) :fire:  
- 10/12/2023: Release [LLM-Embedder](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder), a unified embedding model to support diverse retrieval augmentation needs for LLMs. [Technical Report](https://arxiv.org/pdf/2310.07554.pdf)
- 09/15/2023: The [technical report](https://arxiv.org/pdf/2309.07597.pdf) of BGE has been released 
- 09/15/2023: The [massive training data](https://data.baai.ac.cn/details/BAAI-MTP) of BGE has been released 
- 09/12/2023: New models: 
    - **New reranker model**: release cross-encoder models `BAAI/bge-reranker-base` and `BAAI/bge-reranker-large`, which are more powerful than embedding model. We recommend to use/fine-tune them to re-rank top-k documents returned by embedding models. 
    - **update embedding model**: release `bge-*-v1.5` embedding model to alleviate the issue of the similarity distribution, and enhance its retrieval ability without instruction.
 

<details>
  <summary>More</summary>
<!-- ### More -->
    
- 09/07/2023: Update [fine-tune code](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/README.md): Add script to mine hard negatives and support adding instruction during fine-tuning. 
- 08/09/2023: BGE Models are integrated into **Langchain**, you can use it like [this](#using-langchain); C-MTEB **leaderboard** is [available](https://huggingface.co/spaces/mteb/leaderboard).  
- 08/05/2023: Release base-scale and small-scale models, **best performance among the models of the same size 🤗**  
- 08/02/2023: Release `bge-large-*`(short for BAAI General Embedding) Models, **rank 1st on MTEB and C-MTEB benchmark!** :tada: :tada:   
- 08/01/2023: We release the [Chinese Massive Text Embedding Benchmark](https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB) (**C-MTEB**), consisting of 31 test dataset.  
  
</details>


## Model List

`bge` is short for `BAAI general embedding`.

|              Model              | Language | | Description | query instruction for retrieval [1] |
|:-------------------------------|:--------:| :--------:| :--------:|:--------:|
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)                   |    Multilingual     |    [Inference](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/BGE_M3#usage) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/BGE_M3)    | Multi-Functionality(dense retrieval, sparse retrieval, multi-vector(colbert)), Multi-Linguality, and Multi-Granularity(8192 tokens) |  |
|  [BAAI/llm-embedder](https://huggingface.co/BAAI/llm-embedder)  |   English | [Inference](./FlagEmbedding/llm_embedder/README.md) [Fine-tune](./FlagEmbedding/llm_embedder/README.md) | a unified embedding model to support diverse retrieval augmentation needs for LLMs | See [README](./FlagEmbedding/llm_embedder/README.md) |
|  [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large)  |   Chinese and English | [Inference](#usage-for-reranker) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/reranker) | a cross-encoder model which is more accurate but less efficient [2] |   |
|  [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) |   Chinese and English | [Inference](#usage-for-reranker) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/reranker) | a cross-encoder model which is more accurate but less efficient [2] |   |
|  [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) |   English | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | version 1.5 with more reasonable similarity distribution | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) |   English | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | version 1.5 with more reasonable similarity distribution | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) |   English | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | version 1.5 with more reasonable similarity distribution  | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5) |   Chinese | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | version 1.5 with more reasonable similarity distribution | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5) |   Chinese |  [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | version 1.5 with more reasonable similarity distribution | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) |   Chinese | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | version 1.5 with more reasonable similarity distribution | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-large-en](https://huggingface.co/BAAI/bge-large-en) |   English | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | :trophy: rank **1st** in [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-base-en](https://huggingface.co/BAAI/bge-base-en) |   English | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | a base-scale model but with similar ability to `bge-large-en` | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-small-en](https://huggingface.co/BAAI/bge-small-en) |   English | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |a small-scale model but with competitive performance  | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh) |   Chinese | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | :trophy: rank **1st** in [C-MTEB](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB) benchmark | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-base-zh](https://huggingface.co/BAAI/bge-base-zh) |   Chinese |  [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | a base-scale model but with similar ability to `bge-large-zh` | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh) |   Chinese | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | a small-scale model but with competitive performance | `为这个句子生成表示以用于检索相关文章：`  |

[1\]: If you need to search the relevant passages to a query, we suggest to add the instruction to the query; in other cases, no instruction is needed, just use the original query directly. In all cases, **no instruction** needs to be added to passages.

[2\]: Different from embedding model, reranker uses question and document as input and directly output similarity instead of embedding. To balance the accuracy and time cost, cross-encoder is widely used to re-rank top-k documents retrieved by other simple models. 
For examples, use bge embedding model to retrieve top 100 relevant documents, and then use bge reranker to re-rank the top 100 document to get the final top-3 results.

All models have been uploaded to Huggingface Hub, and you can see them at https://huggingface.co/BAAI. 
If you cannot open the Huggingface Hub, you also can download the models at https://model.baai.ac.cn/models .


## Frequently asked questions

<details>
  <summary>1. How to fine-tune bge embedding model?</summary>

  <!-- ### How to fine-tune bge embedding model? -->
Following this [example](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) to prepare data and fine-tune your model. 
Some suggestions:
- Mine hard negatives following this [example](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune#hard-negatives), which can improve the retrieval performance.
- If you pre-train bge on your data, the pre-trained model cannot be directly used to calculate similarity, and it must be fine-tuned with contrastive learning before computing similarity.
- If the accuracy of the fine-tuned model is still not high, it is recommended to use/fine-tune the cross-encoder model (bge-reranker) to re-rank top-k results. Hard negatives also are needed to fine-tune reranker.

  
</details>

<details>
  <summary>2. The similarity score between two dissimilar sentences is higher than 0.5</summary>

  <!-- ### The similarity score between two dissimilar sentences is higher than 0.5 -->
**Suggest to use bge v1.5, which alleviates the issue of the similarity distribution.** 

Since we finetune the models by contrastive learning with a temperature of 0.01, 
the similarity distribution of the current BGE model is about in the interval \[0.6, 1\].
So a similarity score greater than 0.5 does not indicate that the two sentences are similar.

For downstream tasks, such as passage retrieval or semantic similarity, 
**what matters is the relative order of the scores, not the absolute value.**
If you need to filter similar sentences based on a similarity threshold, 
please select an appropriate similarity threshold based on the similarity distribution on your data (such as 0.8, 0.85, or even 0.9).

</details>

<details>
  <summary>3. When does the query instruction need to be used</summary>

  <!-- ### When does the query instruction need to be used -->

For the `bge-*-v1.5`, we improve its retrieval ability when not using instruction. 
No instruction only has a slight degradation in retrieval performance compared with using instruction. 
So you can generate embedding without instruction in all cases for convenience.
 
For a retrieval task that uses short queries to find long related documents, 
it is recommended to add instructions for these short queries.
**The best method to decide whether to add instructions for queries is choosing the setting that achieves better performance on your task.**
In all cases, the documents/passages do not need to add the instruction. 

</details>


## Usage 

### Usage for Embedding Model

Here are some examples for using `bge` models with 
[FlagEmbedding](#using-flagembedding), [Sentence-Transformers](#using-sentence-transformers), [Langchain](#using-langchain), or [Huggingface Transformers](#using-huggingface-transformers).

#### Using FlagEmbedding
```
pip install -U FlagEmbedding
```
If it doesn't work for you, you can see [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/README.md) for more methods to install FlagEmbedding.

```python
from FlagEmbedding import FlagModel
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
model = FlagModel('BAAI/bge-large-zh-v1.5', 
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
embeddings_1 = model.encode(sentences_1)
embeddings_2 = model.encode(sentences_2)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)

# for s2p(short query to long passage) retrieval task, suggest to use encode_queries() which will automatically add the instruction to each query
# corpus in retrieval task can still use encode() or encode_corpus(), since they don't need instruction
queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
q_embeddings = model.encode_queries(queries)
p_embeddings = model.encode(passages)
scores = q_embeddings @ p_embeddings.T
```
For the value of the argument `query_instruction_for_retrieval`, see [Model List](https://github.com/FlagOpen/FlagEmbedding/tree/master#model-list). 

By default, FlagModel will use all available GPUs when encoding. Please set `os.environ["CUDA_VISIBLE_DEVICES"]` to select specific GPUs.
You also can set `os.environ["CUDA_VISIBLE_DEVICES"]=""` to make all GPUs unavailable.


#### Using Sentence-Transformers

You can also use the `bge` models with [sentence-transformers](https://www.SBERT.net):

```
pip install -U sentence-transformers
```
```python
from sentence_transformers import SentenceTransformer
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
embeddings_1 = model.encode(sentences_1, normalize_embeddings=True)
embeddings_2 = model.encode(sentences_2, normalize_embeddings=True)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
```
For s2p(short query to long passage) retrieval task, 
each short query should start with an instruction (instructions see [Model List](https://github.com/FlagOpen/FlagEmbedding/tree/master#model-list)). 
But the instruction is not needed for passages.
```python
from sentence_transformers import SentenceTransformer
queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
instruction = "为这个句子生成表示以用于检索相关文章："

model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
q_embeddings = model.encode([instruction+q for q in queries], normalize_embeddings=True)
p_embeddings = model.encode(passages, normalize_embeddings=True)
scores = q_embeddings @ p_embeddings.T
```

#### Using Langchain 

You can use `bge` in langchain like this:
```python
from langchain.embeddings import HuggingFaceBgeEmbeddings
model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为这个句子生成表示以用于检索相关文章："
)
model.query_instruction = "为这个句子生成表示以用于检索相关文章："
```


#### Using HuggingFace Transformers

With the transformers package, you can use the model like this: First, you pass your input through the transformer model, then you select the last hidden state of the first token (i.e., [CLS]) as the sentence embedding.

```python
from transformers import AutoTokenizer, AutoModel
import torch
# Sentences we want sentence embeddings for
sentences = ["样例数据-1", "样例数据-2"]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
model.eval()

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
# for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
# encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    sentence_embeddings = model_output[0][:, 0]
# normalize embeddings
sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
print("Sentence embeddings:", sentence_embeddings)
```

### Usage for Reranker

Different from embedding model, reranker uses question and document as input and directly output similarity instead of embedding. 
You can get a relevance score by inputting query and passage to the reranker. 
The reranker is optimized based cross-entropy loss, so the relevance score is not bounded to a specific range.


#### Using FlagEmbedding
```
pip install -U FlagEmbedding
```

Get relevance scores (higher scores indicate more relevance):
```python
from FlagEmbedding import FlagReranker
reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

score = reranker.compute_score(['query', 'passage'])
print(score)

scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']])
print(scores)
```


#### Using Huggingface transformers

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')
model.eval()

pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(scores)
```

#### Usage of the ONNX files

```python
from optimum.onnxruntime import ORTModelForFeatureExtraction  # type: ignore

import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5')
model_ort = ORTModelForFeatureExtraction.from_pretrained('BAAI/bge-small-en-v1.5', file_name="onnx/model.onnx")

# Sentences we want sentence embeddings for
sentences = ["样例数据-1", "样例数据-2"]

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
# for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
# encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

model_output_ort = model_ort(**encoded_input)
# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# model_output and model_output_ort are identical

```

#### Usage via infinity
Its also possible to deploy the onnx files with the [infinity_emb](https://github.com/michaelfeil/infinity) pip package.
Recommended is `device="cuda", engine="torch"` with flash attention on gpu, and `device="cpu", engine="optimum"` for onnx inference.

```python
import asyncio
from infinity_emb import AsyncEmbeddingEngine, EngineArgs

sentences = ["Embed this is sentence via Infinity.", "Paris is in France."]
engine = AsyncEmbeddingEngine.from_args(
    EngineArgs(model_name_or_path = "BAAI/bge-small-en-v1.5", device="cpu", engine="optimum" # or engine="torch"
))

async def main(): 
    async with engine:
        embeddings, usage = await engine.embed(sentences=sentences)
asyncio.run(main())
```


## Evaluation  

`baai-general-embedding` models achieve **state-of-the-art performance on both MTEB and C-MTEB leaderboard!**
For more details and evaluation tools see our [scripts](https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB/README.md). 

- **MTEB**:   

| Model Name |  Dimension | Sequence Length | Average (56) | Retrieval (15) |Clustering (11) | Pair Classification (3) | Reranking (4) |  STS (10) | Summarization (1) | Classification (12) |
|:----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) | 1024 | 512 |  **64.23** | **54.29** |  46.08 | 87.12 | 60.03 | 83.11 | 31.61 | 75.97 |  
| [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) |  768 | 512 | 63.55 | 53.25 |   45.77 | 86.55 | 58.86 | 82.4 | 31.07 | 75.53 |  
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) |  384 | 512 | 62.17 |51.68 | 43.82 |  84.92 | 58.36 | 81.59 | 30.12 | 74.14 |  
| [bge-large-en](https://huggingface.co/BAAI/bge-large-en) |  1024 | 512 | 63.98 |  53.9 | 46.98 | 85.8 | 59.48 | 81.56 | 32.06 | 76.21 | 
| [bge-base-en](https://huggingface.co/BAAI/bge-base-en) |  768 | 512 |  63.36 | 53.0 | 46.32 | 85.86 | 58.7 | 81.84 | 29.27 | 75.27 | 
| [gte-large](https://huggingface.co/thenlper/gte-large) |  1024 | 512 | 63.13 | 52.22 | 46.84 | 85.00 | 59.13 | 83.35 | 31.66 | 73.33 |
| [gte-base](https://huggingface.co/thenlper/gte-base) 	|  768 | 512 | 62.39 | 51.14 | 46.2 | 84.57 | 58.61 | 82.3 | 31.17 | 73.01 |
| [e5-large-v2](https://huggingface.co/intfloat/e5-large-v2) |  1024| 512 | 62.25 | 50.56 | 44.49 | 86.03 | 56.61 | 82.05 | 30.19 | 75.24 |
| [bge-small-en](https://huggingface.co/BAAI/bge-small-en) |  384 | 512 | 62.11 |  51.82 | 44.31 | 83.78 | 57.97 | 80.72 | 30.53 | 74.37 |  
| [instructor-xl](https://huggingface.co/hkunlp/instructor-xl) |  768 | 512 | 61.79 | 49.26 | 44.74 | 86.62 | 57.29 | 83.06 | 32.32 | 61.79 |
| [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) |  768 | 512 | 61.5 | 50.29 | 43.80 | 85.73 | 55.91 | 81.05 | 30.28 | 73.84 |
| [gte-small](https://huggingface.co/thenlper/gte-small) |  384 | 512 | 61.36 | 49.46 | 44.89 | 83.54 | 57.7 | 82.07 | 30.42 | 72.31 |
| [text-embedding-ada-002](https://platform.openai.com/docs/guides/embeddings) | 1536 | 8192 | 60.99 | 49.25 | 45.9 | 84.89 | 56.32 | 80.97 | 30.8 | 70.93 |
| [e5-small-v2](https://huggingface.co/intfloat/e5-base-v2) | 384 | 512 | 59.93 | 49.04 | 39.92 | 84.67 | 54.32 | 80.39 | 31.16 | 72.94 |
| [sentence-t5-xxl](https://huggingface.co/sentence-transformers/sentence-t5-xxl) |  768 | 512 | 59.51 | 42.24 | 43.72 | 85.06 | 56.42 | 82.63 | 30.08 | 73.42 |
| [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) 	|  768 | 514 	| 57.78 | 43.81 | 43.69 | 83.04 | 59.36 | 80.28 | 27.49 | 65.07 |
| [sgpt-bloom-7b1-msmarco](https://huggingface.co/bigscience/sgpt-bloom-7b1-msmarco) 	|  4096 | 2048 | 57.59 | 48.22 | 38.93 | 81.9 | 55.65 | 77.74 | 33.6 | 66.19 |



- **C-MTEB**:  
We create the benchmark C-MTEB for Chinese text embedding which consists of 31 datasets from 6 tasks. 
Please refer to [C_MTEB](https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB/README.md) for a detailed introduction.
 
| Model | Embedding dimension | Avg | Retrieval | STS | PairClassification | Classification | Reranking | Clustering |
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [**BAAI/bge-large-zh-v1.5**](https://huggingface.co/BAAI/bge-large-zh-v1.5) | 1024 |  **64.53** | 70.46 | 56.25 | 81.6 | 69.13 | 65.84 | 48.99 |  
| [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5) | 768 |  63.13 | 69.49 | 53.72 | 79.75 | 68.07 | 65.39 | 47.53 |  
| [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) | 512 | 57.82 | 61.77 | 49.11 | 70.41 | 63.96 | 60.92 | 44.18 |   
| [BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh) | 1024 | 64.20 | 71.53 | 54.98 | 78.94 | 68.32 | 65.11 | 48.39 |
| [bge-large-zh-noinstruct](https://huggingface.co/BAAI/bge-large-zh-noinstruct) | 1024 | 63.53 | 70.55 | 53 | 76.77 | 68.58 | 64.91 | 50.01 |
| [BAAI/bge-base-zh](https://huggingface.co/BAAI/bge-base-zh) | 768 | 62.96 | 69.53 | 54.12 | 77.5 | 67.07 | 64.91 | 47.63 |
| [multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) | 1024 | 58.79 | 63.66 | 48.44 | 69.89 | 67.34 | 56.00 | 48.23 |
| [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh) | 512 | 58.27 |  63.07 | 49.45 | 70.35 | 63.64 | 61.48 | 45.09 |
| [m3e-base](https://huggingface.co/moka-ai/m3e-base) | 768 | 57.10 | 56.91 | 50.47 | 63.99 | 67.52 | 59.34 | 47.68 |
| [m3e-large](https://huggingface.co/moka-ai/m3e-large) | 1024 |  57.05 | 54.75 | 50.42 | 64.3 | 68.2 | 59.66 | 48.88 |
| [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) | 768 | 55.48 | 61.63 | 46.49 | 67.07 | 65.35 | 54.35 | 40.68 |
| [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | 384 | 55.38 | 59.95 | 45.27 | 66.45 | 65.85 | 53.86 | 45.26 |
| [text-embedding-ada-002(OpenAI)](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) | 1536 |  53.02 | 52.0 | 43.35 | 69.56 | 64.31 | 54.28 | 45.68 |
| [luotuo](https://huggingface.co/silk-road/luotuo-bert-medium) | 1024 | 49.37 |  44.4 | 42.78 | 66.62 | 61 | 49.25 | 44.39 |
| [text2vec-base](https://huggingface.co/shibing624/text2vec-base-chinese) | 768 |  47.63 | 38.79 | 43.41 | 67.41 | 62.19 | 49.45 | 37.66 |
| [text2vec-large](https://huggingface.co/GanymedeNil/text2vec-large-chinese) | 1024 | 47.36 | 41.94 | 44.97 | 70.86 | 60.66 | 49.16 | 30.02 |


- **Reranking**:
See [C_MTEB](https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB/) for evaluation script.

| Model | T2Reranking | T2RerankingZh2En\* | T2RerankingEn2Zh\* | MMarcoReranking | CMedQAv1 | CMedQAv2 | Avg |  
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|  
| text2vec-base-multilingual | 64.66 | 62.94 | 62.51 | 14.37 | 48.46 | 48.6 | 50.26 |  
| multilingual-e5-small | 65.62 | 60.94 | 56.41 | 29.91 | 67.26 | 66.54 | 57.78 |  
| multilingual-e5-large | 64.55 | 61.61 | 54.28 | 28.6 | 67.42 | 67.92 | 57.4 |  
| multilingual-e5-base | 64.21 | 62.13 | 54.68 | 29.5 | 66.23 | 66.98 | 57.29 |  
| m3e-base | 66.03 | 62.74 | 56.07 | 17.51 | 77.05 | 76.76 | 59.36 |  
| m3e-large | 66.13 | 62.72 | 56.1 | 16.46 | 77.76 | 78.27 | 59.57 |  
| bge-base-zh-v1.5 | 66.49 | 63.25 | 57.02 | 29.74 | 80.47 | 84.88 | 63.64 |  
| bge-large-zh-v1.5 | 65.74 | 63.39 | 57.03 | 28.74 | 83.45 | 85.44 | 63.97 |  
| [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) | 67.28 | 63.95 | 60.45 | 35.46 | 81.26 | 84.1 | 65.42 |  
| [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) | 67.6 | 64.03 | 61.44 | 37.16 | 82.15 | 84.18 | 66.09 |  

\* : T2RerankingZh2En and T2RerankingEn2Zh are cross-language retrieval tasks

## Train

### BAAI Embedding 

We pre-train the models using [retromae](https://github.com/staoxiao/RetroMAE) and train them on large-scale pairs data using contrastive learning. 
**You can fine-tune the embedding model on your data following our [examples](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune).**
We also provide a [pre-train example](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/pretrain).
Note that the goal of pre-training is to reconstruct the text, and the pre-trained model cannot be used for similarity calculation directly, it needs to be fine-tuned.
More training details for bge see [baai_general_embedding](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/README.md).



### BGE Reranker

Cross-encoder will perform full-attention over the input pair, 
which is more accurate than embedding model (i.e., bi-encoder) but more time-consuming than embedding model.
Therefore, it can be used to re-rank the top-k documents returned by embedding model.
We train the cross-encoder on a multilingual pair data, 
The data format is the same as embedding model, so you can fine-tune it easily following our [example](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/reranker). 
More details please refer to [./FlagEmbedding/reranker/README.md](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker)


## Contact
If you have any question or suggestion related to this project, feel free to open an issue or pull request.
You also can email Shitao Xiao(stxiao@baai.ac.cn) and Zheng Liu(liuzheng@baai.ac.cn). 


## Citation

If you find this repository useful, please consider giving a star :star: and citation

```
@misc{bge_embedding,
      title={C-Pack: Packaged Resources To Advance General Chinese Embedding}, 
      author={Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff},
      year={2023},
      eprint={2309.07597},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License
FlagEmbedding is licensed under the [MIT License](https://github.com/FlagOpen/FlagEmbedding/blob/master/LICENSE). The released models can be used for commercial purposes free of charge.

