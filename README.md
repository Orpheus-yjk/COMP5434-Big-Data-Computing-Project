# COMP5434-Big-Data-Computing-Project
**The primary objectives of this project are**:
 1. Analyze the CORD-19 dataset to understand the evolving landscape of COVID-19 research.
 2. Identify key areas of research focus within the dataset.
 3. Discover relationships and connections between different research topics.
 4. Generate insights that can inform future research directions and collaborations
# DataPreparation
 • Load the CORD-19 (subset) dataset from “meta_10k.csv” and “subset.zip” (json files), and print some meta infor
mationsuch as data length, the first n rows of the data.

 • Apply text cleaning techniques to remove irrelevant characters, stop words [5], and punctuation.
 
 • Performsomeexploratoryanalysis: distribution of languages, histogram of publication years (showing the number
 of papers per year), and histogram of journals (showing the number of papers per journal)


# MapReduce
 Your assignment entails the development of a program, utilizing the MapReduce paradigm, for the purpose of word
 counting and index building within the dataset. You should:
 • Write aMapReduce-style program to count the frequency of words.
 
 • Construct a stop word list to exclude meaningless words, and return the top-50 prevalent words.
 
 • Write a MapReduce-style program to generate an index, which can facilitate the retrieval of document ids to the
 queried words.
 
 • Youcanengageiniterative refinement of the stop word list to count domain-specific words, still return the top-50
 words.


# Association Analysis
 • Apply frequent pattern mining algorithms (e.g., Apriori, FP-Growth) to the processed document representations.
 
 • Identify frequent co-occurring terms and analyze their significance within the context.
 
 • Analyze relationships between discovered topics and research trends.
 
 • Exploretopicmodelingtechniques(e.g.,LatentDirichletallocation(LDA))toidentifylatenttopicswithinthedataset.


# Similarity Analysis
 • Choosefeasible distance metric to measure the similarity between documents.
 
 • Implement Locality-Sensitive Hashing (LSH) to find similar research papers based on their document representa
tions.

 • Experiment with different LSH families and parameters to optimize similarity calculations.
 
 • Explore methods to account for semantic similarity using word embedding.


# Clustering Analysis
 • Apply clustering algorithms (e.g., k-means, DBSCAN, hierarchical clustering) to group research papers based on
 their topics, publication dates, or other relevant features.
 
 • Experiment with different algorithms and feature engineering techniques to improve clustering performance.
 
 • Analyze the characteristics of each research cluster and visualize the results


# Tools and Technologies
 This project is implemented using the following tools:
 • Scarfold: Python, PySpark
 
 • Computation: NumPy
 
 • DataProcessing: Pandas
 
 • DataMining Algorithms: scikit-learn
 
 • NLP tools: NLTK, spaCy(tokenization, tagging, parsing, named entity recognition, text classification and
 more)
 • Visualization: Matplotlib, seaborn


