# About
This is the code developed for my capstone project of TI4141 Data Analytics course, Industrial Engineering @ Institut Teknologi Bandung, Indonesia. This code is for academic purpose only and the report paper is not to be published whatsoever. The project purpose is to develop a customer segmentation analysis for a houseware retail network in Indonesia. Customer segmentation is done with RFM analysis, which considers customers' recency, frequency, and monetary value to the network in doing segmentation. K-Means Clustering algorithm is used for the data-based segmentation. For resource and legal reasons I would not provide the datasets used for this project.

I included the notebook file in case you want to view the result direcly from the github page

# Result
![Cluster Visualization](https://drive.google.com/uc?id=1AoK2TypK4LFE_cBtixN5vYWSsCv6iQqC)
## Cluster Profile
Cluster 0: Highest freq and spending, with the lowest recency (most recent last transactions)
Cluster 1: Lowest freq and spending, highest recency (oldest last transactions)
Cluster 2: Intermediate (just between the 0 and 1 cluster)

## Business Implication
Cluster 0 = Loyal Customer --> give upmost priority and special treatment, avoid churn at every cost
Cluster 1 = Lost Customer / Bypasser --> They only transact every once in a while, no need to spend resource for this cluster
Cluster 2 = Opportunity --> Design marketing strategies to increase their value to the supermarket
