#!/usr/bin/env python
# coding: utf-8

# # About the Dataset
# 
# The original CORD-19 is a resource of over 1,000,000 scholarly articles, including over 400,000 with full text, about COVID-19, SARS-CoV-2, and related coronaviruses.
# 
# In our project, the dataset is sampled from the CORD-19 with size ~10,000 to reduce computation burden.

# In[1]:


import os
import subprocess


# shared link: https://drive.google.com/drive/folders/1Td_ZTUVrsKeftDE5Zll7252YLJdWiNTk?usp=share_link
# you can download the data via the shared link, and skip Step 0 and Step 1 if you want to run the code in your local machine


# # Step 0: add the shared folder to your google drive. e.g., /content/drive/MyDrive/CORD_19

# # Step 1: Mount Google Drive
# from google.colab import drive
# drive.mount("/content/drive", force_remount=True)


# !echo $PWD

# !ls /content/drive/MyDrive/CORD_19/

# # Step 2: unzip json files
# data_root = os.path.join(os.getcwd(),"CORD_19")
# subset_dir = os.path.join(data_root,"CORD_19_subset")

# zip_file_path=os.path.join(data_root,"subset.zip")

# print("zip_file_path:",zip_file_path)
# print("subset_dir:",subset_dir)

# # Check if the destination directory exists
# if not os.path.exists(subset_dir):
#     # Unzip the file
#     print("OS FALSE? ",os.system("mkdir \"{}/\"".format(subset_dir)))
#     cmd = "unzip \"{}\" -d \"{}\"".format(zip_file_path, subset_dir)
#     print("CMD: ",cmd)
#     proc = subprocess.Popen(cmd, shell=True)
# else:
#     print(f"Directory {subset_dir} already exists. Skipping extraction.")


# In[2]:


# import packages

import os
import json
import glob
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# # Data Load & Pre-processing

# In[3]:


import time
STARTTIME=time.time()


# In[4]:


import warnings
warnings.filterwarnings("ignore")


# In[5]:


# Load Meta data from meta_10k.csv
data_root = os.path.join(os.getcwd(),"CORD_19")
metadata_path = os.path.join(data_root, 'meta_10k.csv')

meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str,
    'doi': str
})

print(len(meta_df))
meta_df.head()


# In[6]:


meta_df.info()


# In[7]:


def glob_files(path, f_type=".json"):
    dst = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(f_type):
                dst.append(os.path.join(root, f))
    return dst

# glob json files
subset_dir = os.path.join(data_root,"CORD_19_subset")
json_dir = os.path.join(subset_dir, "subset\document_parses\pdf_json")
print(json_dir)
json_files = glob_files(json_dir, ".json")

print("total json files:", len(json_files))


# In[8]:


class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            # Abstract
            for entry in content['abstract']:
                self.abstract.append(entry['text'])
            # Body text
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)

            self.title = content['metadata']['title']

            #dict_keys(['paper_id', 'metadata', 'abstract', 'body_text',
            #'bib_entries', 'ref_entries', 'back_matter'])


    def __repr__(self):
        return f"{self.paper_id}: {self.title } : {self.abstract[:200]}... {self.body_text[:200]}..."


first_row = FileReader(json_files[0])
print(first_row)


# In[9]:


from tqdm import tqdm

def get_breaks(content, length):
    data = ""
    words = content.split(' ')
    total_chars = 0

    # add break every length characters
    for i in range(len(words)):
        total_chars += len(words[i])
        if total_chars > length:
            data = data + "<br>" + words[i]
            total_chars = 0
        else:
            data = data + " " + words[i]
    return data


dict_ = {'paper_id': [], 'doi':[], 'abstract': [], 'body_text': [],
         'authors': [], 'title': [], 'journal': [], 'abstract_summary': []}


for idx, entry in tqdm(enumerate(json_files), total=len(json_files)):
    try:
        content = FileReader(entry)
    except Exception as e:
        continue  # invalid paper format, skip

    # get metadata information
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
    # no metadata, skip this paper
    if len(meta_data) == 0:
        continue
    if len(content.body_text) == 0:
        continue
    dict_['abstract'].append(content.abstract)
    dict_['paper_id'].append(content.paper_id)
    dict_['body_text'].append(content.body_text)
    # also create a column for the summary of abstract to be used in a plot
    if len(content.abstract) == 0:
        # no abstract provided
        dict_['abstract_summary'].append("Not provided.")
    elif len(content.abstract.split(' ')) > 100:
        # abstract provided is too long for plot, take first 300 words append with ...
        info = content.abstract.split(' ')[:100]
        summary = get_breaks(' '.join(info), 40)
        dict_['abstract_summary'].append(summary + "...")
    else:
        # abstract is short enough
        summary = get_breaks(content.abstract, 40)
        dict_['abstract_summary'].append(summary)

    # get metadata information
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

    try:
        # if more than one author
        authors = meta_data['authors'].values[0].split(';')
        if len(authors) > 2:
            # more than 2 authors, may be problem when plotting, so take first 2 append with ...
            dict_['authors'].append(get_breaks('. '.join(authors), 40))
        else:
            # authors will fit in plot
            dict_['authors'].append(". ".join(authors))
    except Exception as e:
        # if only one author - or Null valie
        dict_['authors'].append(meta_data['authors'].values[0])

    # add the title information, add breaks when needed
    try:
        title = get_breaks(meta_data['title'].values[0], 40)
        dict_['title'].append(title)
    # if title was not provided
    except Exception as e:
        dict_['title'].append(meta_data['title'].values[0])

    # add the journal information
    dict_['journal'].append(meta_data['journal'].values[0])

    # add doi
    dict_['doi'].append(meta_data['doi'].values[0])


df_covid = pd.DataFrame(dict_, columns=['paper_id', 'doi', 'abstract', 'body_text',
                                        'authors', 'title', 'journal', 'abstract_summary'])
df_covid.head()


# In[10]:


df_covid['publish_time']=meta_df['publish_time'].copy()


# In[11]:


df_covid.info()


# In[12]:


df = df_covid
df.dropna(inplace=True)
df.info()


# In[13]:


get_ipython().system('pip install langdetect')


# In[14]:


from tqdm import tqdm
from langdetect import detect
from langdetect import DetectorFactory

# set seed
DetectorFactory.seed = 0

# hold label - language
languages = []

# go through each text
for ii in tqdm(range(0,len(df))):
    # split by space into list, take the first x intex, join with space
    text = df.iloc[ii]['body_text'].split(" ")

    lang = "en"
    try:
        if len(text) > 50:
            lang = detect(" ".join(text[:50]))
        elif len(text) > 0:
            lang = detect(" ".join(text[:len(text)]))
    # ught... beginning of the document was not in a good format
    except Exception as e:
        all_words = set(text)
        try:
            lang = detect(" ".join(all_words))
        # what!! :( let's see if we can find any text in abstract...
        except Exception as e:

            try:
                # let's try to label it through the abstract then
                lang = detect(df.iloc[ii]['abstract_summary'])
            except Exception as e:
                lang = "unknown"
                pass

    # get the language
    languages.append(lang)


# In[15]:


from pprint import pprint

languages_dict = {}
for lang in set(languages):
    languages_dict[lang] = languages.count(lang)

print("Total: {}\n".format(len(languages)))
pprint(languages_dict)


# In[16]:


df['language'] = languages
df = df[df['language'] == 'en']
df.info()


# # Histogram of year / journal

# In[17]:


#Calculate the total count
total_count=sum(languages_dict.values())
# Extract languages and their corresponding counts
languages =list(languages_dict.keys())
counts =list(languages_dict.values())
# Calculate frequencies
frequencies =[count /total_count for count in counts]
# Create the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(languages, counts, color='skyblue')
# Add title and labels
plt.title('Language Distribution')
plt.xlabel('Languages')
plt.ylabel('Counts')
# Display count and frequency labels on the bars
for bar, count, freq in zip(bars, counts, frequencies):
    height = bar.get_height()
    plt.text(bar.get_x()+ bar.get_width()/2,height + 50, f'{count}\n({freq:2%})', ha='center')
# Show the bar chart
plt.show()


# In[18]:


import pandas as pd
import matplotlib.pyplot as plt

meta_df['publish_time_yr'] = pd.to_datetime(meta_df['publish_time'], errors='coerce').dt.year

# Calculate the number of papers per year
paper_count = meta_df['publish_time_yr'].value_counts().sort_index()

# Plotting bar charts
plt.bar(paper_count.index, paper_count.values)

# Setting the chart title and axis labels
plt.title('Histogram of year of publication')
plt.xlabel('Year of publication')
plt.ylabel('Number of papers')

# Show charts
plt.show()


# In[19]:


import matplotlib.pyplot as plt

# Calculate the number of papers per journal
journal_count = meta_df['journal'].value_counts()

journal_count0 = journal_count[journal_count.values>=30]
# Plotting bar charts
plt.bar(journal_count0.index, journal_count0.values)
# Setting the chart title and axis labels
plt.title('Journal Bar Chart(journal_count.value>30)')
plt.xlabel('periodicals')
plt.ylabel('Journal Name')
# In order to avoid overlapping horizontal coordinate text, adjustments can be made using the following code
plt.xticks(rotation=90)
# Show charts
plt.show()


journal_count0 = journal_count[journal_count.values<30]
journal_count0 = journal_count0[journal_count0.values>=20]
# Plotting bar charts
plt.bar(journal_count0.index, journal_count0.values)
# Setting the chart title and axis labels
plt.title('Journal Bar Chart(20<=journal_count.value<30)')
plt.xlabel('periodicals')
plt.ylabel('Journal Name')

# In order to avoid overlapping horizontal coordinate text, adjustments can be made using the following code
plt.xticks(rotation=90)

# Show charts
plt.show()


# In[20]:


# Histogram cancellation variable:

total_count=languages=counts=frequencies=None
bars=paper_count=journal_count=journal_count0=None


# # Map-Reduce 

# ## Map-Reduce with stop list

# In[21]:


data = pd.DataFrame(dict_, columns=['body_text',
                                        'title','paper_id' ])
def clean_word(data):
    return re.sub(r'[^\w\s]','',data).lower()

from collections import Counter    
import re     

def find_top_words(data_pd):
    cnt = Counter()
    for col in data_pd.columns: #columns=['abstract', 'body_text','authors', 'title', 'journal']
        material_list = data_pd[col]
        print("COL",col)
        for text in material_list:  #paper 0-9021 each column
            try:
                tokens_in_text = text.split()
                tokens_in_text = map(clean_word, tokens_in_text)
                cnt.update(tokens_in_text)
            except AttributeError: # AttributeError: 'float' object has no attribtuion 'spilt'
                return cnt.most_common(50)


# In[22]:


get_ipython().run_line_magic('time', 'find_top_words(data)')


# In[23]:


import nltk
import pandas as pd
nltk.download('stopwords')# Create a set of stop words 
from nltk.corpus import stopwords
 
stop_words = nltk.corpus.stopwords.words('english')
# entend()function is used to add custom stopwords 
stop_list = ["1","2","3","4","5","6","7","8","9","et","al","also","used","using","one","however","n","may","","two","different","p","10",'however',"figure","table","could","fig",]
stop_words.extend(stop_list)
print(stop_words)

data = pd.DataFrame(dict_, columns=[ 'abstract', 'body_text', 'title', 'journal'])
def clean_word(data):
    return re.sub(r'[^\w\s]','',data).lower()

from collections import Counter    
import re     

def find_top_words(data_pd):
    cnt = Counter()
    for col in data_pd.columns: #columns=['abstract', 'body_text','authors', 'title', 'journal']
        material_list = data_pd[col]
        print("COL",col)
        for text in material_list:  #paper 0-9021 each column
            try:
                tokens_in_text = text.split()
                tokens_in_text = [w for w in tokens_in_text if not w.lower() in stop_words]
                tokens_in_text = map(clean_word, tokens_in_text)
                cnt.update(tokens_in_text)
            except AttributeError: # AttributeError: 'float' object has no attribtuion 'spilt'
                return cnt.most_common(50)
        


# In[24]:


from collections import Counter
import re
import concurrent.futures

# Function to clean a word
def clean_word(word):
    return re.sub(r'[^\w\s]', '', word).lower()

# Function to perform the Word Count MapReduce
def word_count_map(text, stop_words):
    word_counts = Counter()
    words = text.split()
    for word in words:
        cleaned_word = clean_word(word)
        if cleaned_word and cleaned_word not in stop_words:
            word_counts[cleaned_word] += 1
    return word_counts

# Function to combine word counts from different mappers
def combine_word_counts(results):
    combined_word_counts = Counter()
    for word_count in results:
        combined_word_counts.update(word_count)
    return combined_word_counts

# Define the stop word list
import nltk
nltk.download('stopwords')# Create a set of stop words 
from nltk.corpus import stopwords
 
stop_words = nltk.corpus.stopwords.words('english')
# entend()function is used to add custom stopwords 
stop_list = ["1","2","3","4","5","6","7","8","9","et","al","also","used","using","one","however","n","may","","two","different","p","10",'however',"figure","table","could","fig",]
stop_words.extend(stop_list)

# Call the parallel Word Count MapReduce function
word_counts = Counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
    # Map phase
    futures = [executor.submit(word_count_map, text, stop_words)
               for col in data.columns for text in data[col] if isinstance(text, str)]

    # Reduce phase
    results = [future.result() for future in concurrent.futures.as_completed(futures)]
    word_counts = combine_word_counts(results)

# Get the top 50 words
top_50_words = word_counts.most_common(50)

# Print the top 50 most common words
for word, count in top_50_words:
    print(f"{word}: {count}")


# ## Map-Reduce of Paper Id with respective occurances

# In[ ]:





# In[25]:


from collections import Counter
import re
import concurrent.futures
import pandas as pd

# Function to clean a word
def clean_word(word):
    return re.sub(r'[^\w\s]', '', word).lower()

# Function to perform the Word Count MapReduce
def word_count_map(text, stop_words, queried_words, paper_id):
    word_counts = Counter()
    words = text.split()
    for word in words:
        cleaned_word = clean_word(word)
        if cleaned_word and cleaned_word not in stop_words and cleaned_word in queried_words:
            word_counts[cleaned_word] += 1
    return paper_id, word_counts

# Function to combine word counts from different mappers
def combine_word_counts(results):
    combined_word_counts = Counter()
    for _, word_count in results:
        combined_word_counts.update(word_count)
    return combined_word_counts

# Define the stop word list
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = nltk.corpus.stopwords.words('english')
stop_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "et", "al", "also", "used", "using", "one", "however", "n", "may", "", "two", "different", "p", "10", "however", "figure", "table", "could", "fig"]
stop_words.extend(stop_list)

# Define the DataFrame
data = pd.DataFrame(dict_, columns=['abstract', 'body_text', 'authors', 'title', 'journal', 'paper_id'])

# Get the queried words from user input
# queried_words = input("Enter the queried words (separated by spaces): ").split()
# e.g
queried_words = "patients covid19 study health results infection".split()

# Call the parallel Word Count MapReduce function
word_counts = Counter()
index = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
    # Map phase
    futures = []
    for col in data.columns:
        for i, text in enumerate(data[col]):
            if isinstance(text, str):
                paper_id = f"{data.at[i, 'paper_id']}_{data.at[i, 'title']}"
                futures.append(executor.submit(word_count_map, text, stop_words, queried_words, paper_id))

    # Reduce phase
    results = [future.result() for future in concurrent.futures.as_completed(futures)]
    word_counts = combine_word_counts(results)

    # Build index
    for paper_id, word_count in results:
        for word, count in word_count.items():
            if word in index:
                index[word].append((paper_id, count))
            else:
                index[word] = [(paper_id, count)]

# Generate the summary of paper IDs with summed occurrence
summary = {}
for word in queried_words:
    if word in index:
        summary[word] = {
            'Total Occurrences': sum(count for _, count in index[word]),
            'Paper IDs': [(paper_id, count) for paper_id, count in index[word]]
        }
    elif word in stop_words:
        summary[word] = {
            'Total Occurrences': 0,
            'Paper IDs': "In the stop list"
        }
    else:
        summary[word] = {
            'Total Occurrences': 0,
            'Paper IDs': "Not occurred in the dataset"
        }

# Print the summary
for word, info in summary.items():
    print(f"The Queried word is {word}:")
    print(f"  Total Occurrences: {info['Total Occurrences']}")
    if isinstance(info['Paper IDs'], str):
        print(f"  {info['Paper IDs']}")
    else:
        print(f"                   Paper IDs:                           Topics")
        for paper_id, count in info['Paper IDs']:
            print(f"    {paper_id} (Occurrences: {count})")
    print()


# In[26]:


# MapReduce cancellation variable:

data=stop_words=stop_list=word_counts=None
top_50_words=queried_words=index=summary=None


# # Association Analysis

# ## Data Clean

# In[27]:


# !python -m pip install -U pydantic spacy
get_ipython().system('pip install spacy==3.7.5')
get_ipython().system('python -m spacy download en_core_web_sm')


# In[28]:


import spacy
import pandas as pd
import re
from bs4 import BeautifulSoup


# In[29]:


# Removal of special characters, punctuation, html label and numbers.
def clean_text(text):
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text(separator=" ")
    cleaned_text = re.sub(r'[^A-Za-z\s]', '', cleaned_text)
    return cleaned_text

# NLP processing functions, including word splitting, stop word removal
def spacy_process_text(text):
    doc = nlp(text)
    # Removal of stop words using list derivation
    result = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and not token.is_space and token.is_alpha]
    return " ".join(result)


# In[30]:


df3_3 = df.copy(deep=True)
nlp = spacy.load('en_core_web_sm')

# defined custom stop words list
my_stop_words = ['doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure',
    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.',
    'al.', 'Elsevier', 'PMC', 'CZI', ' ', '\n', '  ', '+', ' \n', '  \n']
for stopword in my_stop_words:
    nlp.vocab[stopword].is_stop = True


tqdm.pandas()
df3_3['cleaned_abstract'] = df3_3['abstract'].progress_apply(clean_text)
df3_3['processed_abstract'] = df3_3['cleaned_abstract'].progress_apply(spacy_process_text)

df3_3['cleaned_text'] = df3_3['body_text'].progress_apply(clean_text)
df3_3['processed_text'] = df3_3['cleaned_text'].progress_apply(spacy_process_text)

print(df3_3[['processed_abstract']])
print(df3_3[['processed_text']])

df3_3.to_csv('output.csv', index=False)


# ## TF-IDF

# In[31]:


from sklearn.feature_extraction.text import TfidfVectorizer

process_text_value = df3_3['processed_text'].values
vectorizer = TfidfVectorizer(max_features=2**12)
X = vectorizer.fit_transform(process_text_value)
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

print(tfidf_df)


# In[32]:


tfidf_threshold = 0.07
transactions = []

for index, row in tqdm(tfidf_df.iterrows(), total=tfidf_df.shape[0], desc="Generating Transactions"):
    transactions.append([vectorizer.get_feature_names_out()[i] for i in range(len(row)) if row[i] >= tfidf_threshold])


# ## Applying Fp-growth

# In[33]:


get_ipython().system('pip install wordcloud')
get_ipython().system('pip install networkx matplotlib')
get_ipython().system('pip install pyvis')


# In[34]:


get_ipython().system('pip install mlxtend')


# In[35]:


from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_fpgrowth = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = fpgrowth(df_fpgrowth, min_support=0.01, use_colnames=True).sort_values(by='support', ascending=False)

print(frequent_itemsets)

frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets_cuttoff = frequent_itemsets[frequent_itemsets['length'] > 1]
frequent_itemsets_cuttoff = frequent_itemsets_cuttoff[frequent_itemsets_cuttoff['support'] > 0.03]

print(frequent_itemsets_cuttoff[['itemsets', 'support']])


# In[36]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_word_cloud_vid_itemsets(itemsets):
  # Create word frequency dictionaries for use in word clouds
  word_freq = {}
  for index, row in itemsets.iterrows():
      # Convert the itemset into a single string, separating the items by spaces
      itemset_str = ' '.join(row['itemsets'])
      # Support as frequency
      word_freq[itemset_str] = row['support']

  wordcloud = WordCloud(width=800, height=400, background_color='white')

  # Generate word clouds based on the support of frequent item sets
  wordcloud.generate_from_frequencies(word_freq)

  # Mapping the word cloud
  plt.figure(figsize=(10, 5))
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis('off')
  plt.show()

generate_word_cloud_vid_itemsets(frequent_itemsets)
print("\n")
generate_word_cloud_vid_itemsets(frequent_itemsets_cuttoff)


# In[37]:


rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
filtered_rules = rules[(rules['support'] > 0.010) & (rules['confidence'] > 0.5) & (rules['lift'] > 6)]
print(filtered_rules)


# In[38]:


from pyvis.network import Network

# Create a pyvis network
nt = Network(height='3000px', width='3000px', directed=True, bgcolor="#ffffff", font_color="black")

# Add nodes and edges
for index, row in filtered_rules.iterrows():
    from_node = ', '.join(list(row['antecedents']))
    to_node = ', '.join(list(row['consequents']))
    support = row['support']
    confidence = row['confidence']

    nt.add_node(from_node, title=from_node, size=30)
    nt.add_node(to_node, title=to_node, size=30)
    nt.add_edge(from_node, to_node, title=f'Support: {support:.3f}, Confidence: {confidence:.3f}')

# Set options for network diagrams (using pyvis presets)
network_options = """
{
  "nodes": {
    "font": {
      "size": 22,
      "strokeWidth": 0,
      "color": "#ffffff"
    },
    "scaling": {
      "min": 10,
      "max": 30
    }
  },
  "edges": {
    "smooth": false,
    "color": {
      "inherit": false,
      "color": "#ffffff"
    },
    "font": {
      "size": 22,
      "align": "top"
    }
  },
  "physics": {
    "forceAtlas2Based": {
      "gravitationalConstant": -26,
      "centralGravity": 0.005,
      "springLength": 230,
      "avoidOverlap": 0.0
    },
    "maxVelocity": 146,
    "solver": "forceAtlas2Based",
    "timestep": 0.35,
    "stabilization": {
      "iterations": 150
    }
  },
  "interaction": {
    "dragNodes": true,
    "tooltipDelay": 200,
    "hideEdgesOnDrag": false
  },
  "manipulation": {
    "enabled": true
  }
}
"""

# Save it as an HTML file, but don't open it automatically
nt.save_graph('network.html')


# ## LDA

# In[39]:


from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime

def parse_date(date_str):
    # Define different date formats
    date_formats = [
        "%Y-%m-%d",     # ISO 8601 format
        "%d-%m-%Y",     # European DD-MM-YYYY format
        "%m-%d-%Y",     # US MM-DD-YYYY format
        "%Y/%m/%d",     # Alternate separator format
        "%Y",           # Year only
    ]

    # Iterate through all the date formats and try to parse them
    for date_format in date_formats:
        try:
            return pd.to_datetime(datetime.strptime(date_str, date_format))
        except ValueError:
            pass

    # If it gets here, it means there is no match in the format defined above, try another method or return the default value.
    # Check to see if only the year is included
    if re.match(r'^\d{4}$', date_str):
        # Year only, giving default month of June
        return pd.to_datetime(date_str + '-06-01')
    else:
        # If none of them match, return NaT or other customized value
        return pd.NaT

df3_3['uniform_time'] = meta_df['publish_time'].apply(parse_date)


print(df3_3[['publish_time', 'uniform_time']])

invalid_time_df = df3_3[df3_3['uniform_time'].isna()]

print(f"Number of rows with invalid time: {invalid_time_df.shape[0]}")
print(invalid_time_df[['publish_time', 'uniform_time']])


# In[40]:


from gensim import corpora, models

df3_3_cutoff_timme = df3_3[(df3_3['uniform_time'] >= '2018-01-01') & (df3_3['uniform_time'] <= '2024-12-31')]


# Create dictionaries and corpora
df3_3_cutoff_timme['processed_text_array'] = df3_3_cutoff_timme['processed_text'].map(lambda text: text.split())
dictionary = corpora.Dictionary(df3_3_cutoff_timme["processed_text_array"])
corpus = [dictionary.doc2bow(text) for text in df3_3_cutoff_timme["processed_text_array"]]

# LDA models
num_topics = 5  # topics selected based on dataset
lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=3)


topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)


# In[41]:


df3_3_cutoff_timme['year_month'] = df3_3_cutoff_timme['uniform_time'].dt.to_period('M')

# Get the dominant topic for each document
df3_3_cutoff_timme['topics'] = [lda_model.get_document_topics(bow) for bow in corpus]
df3_3_cutoff_timme['dominant_topic'] = df3_3_cutoff_timme['topics'].apply(lambda topics: max(topics, key=lambda item: item[1])[0])

topic_descriptions = {i: ' '.join([word for word, prob in lda_model.show_topic(i, topn=3)])
                      for i in range(lda_model.num_topics)}
df3_3_cutoff_timme['dominant_topic_label'] = df3_3_cutoff_timme['dominant_topic'].map(topic_descriptions)

# documents by year and leading topic
trend_df = df3_3_cutoff_timme.groupby(['year_month', 'dominant_topic_label']).size().reset_index(name='document_count')
trend_df['year_month'] = trend_df['year_month'].astype(str)


# In[42]:


# Trends in visualization
import seaborn as sns

plt.figure(figsize=(20, 9))
sns.lineplot(data=trend_df, x='year_month', y='document_count', hue='dominant_topic_label')
plt.title('Trends of Topics Over Time')
plt.xlabel('Year-Month')
plt.ylabel('Number of Documents')

plt.xticks(rotation=45)

ax = plt.gca()
labels = ax.get_xticklabels()

for i, label in enumerate(labels):
    if i % 6 != 0:   # Displayed every six months
        label.set_visible(False)

plt.show()


# In[43]:


# Association Analysis cancellation variable:

df3_3=nlp=my_stop_words=process_text_value=None
vectorizer=X=tfidf_df=transactions=0
te=te_ary=df_fpgrowth=frequent_itemsets=None
rules=filtered_rules=nt=None
invalid_time_df=df3_3_cutoff_timme=dictionary=corpus=None
lda_model=trend_df=None


# # Similarity Analysis

# In[44]:


# Reassign a df
# df3_4 = df.copy(deep=True)
df3_4 = df # direct ref
df3_4.columns


# In[45]:


# Text length distribution
sample_column_name = ["body_text", "abstract", "abstract_summary", "authors", "title"]
sample_column_name = df3_4.columns
print(df3_4)
for _sample_column in sample_column_name:
    print(_sample_column)
    print("Total Len of {}:".format(_sample_column), sum([len(text) for text in df3_4[_sample_column]]),
          " ==========  Average Len: ", sum([len(text) for text in df3_4[_sample_column]]) / len(df3_4[_sample_column]) )

# len: body_text > abstract > abstract_summary > authors > title


# ## Explore the text length distribution

# In[46]:


sample_column_name = ["body_text", "abstract", "abstract_summary", "authors", "title"]

import matplotlib.pyplot as plt

# Plotting each subgraph
for i in range(5):
    _sample_column = sample_column_name[i]
    # Calculate text length
    text_lengths = [len(text) for text in df3_4[_sample_column]]
    
    for j in range(3):
        if j==0:
            # Plotting histograms
            plt.hist(text_lengths, bins=100, edgecolor='black')
            # Setting the title and axis labels
            plt.title('Text Length Distribution of {}'.format(_sample_column))
            plt.xlabel('Text Length')
            plt.ylabel('Frequency')
        
        elif j==1:
            # Plotting Box Lines
            plt.boxplot(text_lengths)
            # Setting the title and axis labels
            plt.title('Text Length Distribution of {}'.format(_sample_column))
            plt.ylabel('Text Length')
            # Get details of the box plot
            statistics = plt.boxplot(text_lengths)['medians'][0].get_ydata()
            median = statistics[0]
            q1 = plt.boxplot(text_lengths)['boxes'][0].get_ydata()[0]
            q3 = plt.boxplot(text_lengths)['boxes'][0].get_ydata()[2]
            # Output details
            print("boxplot Info of {}".format(_sample_column))
            print("Median: ", median)
            print("Q1: ", q1)
            print("Q3: ", q3)
        
        elif j==2:
            # Calculate the cumulative distribution function
            sorted_lengths = np.sort(text_lengths)
            cumulative_prob = np.arange(len(sorted_lengths)) / float(len(sorted_lengths))
            # Mapping of cumulative distribution
            plt.plot(sorted_lengths, cumulative_prob, marker='o')
            # Setting the title and axis labels
            plt.title('Cumulative Distribution Plot of {}'.format(_sample_column))
            plt.xlabel('Text Length')
            plt.ylabel('Cumulative Probability')    

        # Display Graphics
        plt.show()


# In[47]:


# length
# body_text > abstract >= abstract_summary >= authors, title


# ## Distance metrics used to measure text similarity

# ### Jaccard implementation

# In[48]:


# When measuring the similarity between two texts, the Jaccard coefficient can be used to calculate how similar they are.
def jaccard(text1, text2):
    set1 = set(build_shingles(text1,K)) # set(text1.lower().split())
    set2 = set(build_shingles(text2,K)) # set(text2.lower().split())
    # Calculate the length of the intersection intersection and the union union of two sets. 
    # Finally we use the ratio of intersection to union to compute the Jaccard coefficient
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union==0:return 1 # Two empty sets
    jaccard_coefficient = intersection / union
    return jaccard_coefficient


# ### splitwords() as an alternative

# In[49]:


# A word embedding method that is much faster than shingle memory and time
def split_into_word_set(text):
    text = text + " "
    wnd = ""
    word_list = []
    for _i in range(len(text)):
        if (text[_i]>="A" and text[_i]<="Z") or (text[_i]>="a" and text[_i]<="z") or (text[_i]>="0" and text[_i]<="9"):
            # Char or Number limited
            wnd = wnd + text[_i]
        else:
            _threshold = 3 # leave out meaning less  word
            if len(wnd)>=_threshold:
                word_list.append(wnd)
            wnd = ""
    return set(word_list)


# In[50]:


# preprocessing Indexs have lost and exists.
sets_rec = []
Idx_rec = []
err_rec = []
for i in range(len(df3_4["abstract"])):
    try:
        sets_rec.append(split_into_word_set(df3_4["abstract"][i]))
        Idx_rec.append(i)
    except KeyError:
        sets_rec.append(set([]))
        err_rec.append(i)
print("{} Indexs have lost. ".format(len(err_rec)))
print("{} Indexs exists. ".format(len(Idx_rec)))  # Even if the index exists, the content may be empty

tot_hascontent = tot_nocontent = 0
for idx in Idx_rec:
    if len(df3_4["abstract"][idx]) == 0:
        tot_nocontent +=1
    else:
        tot_hascontent+=1
        
print()
print("{} Indexs have content. ".format(tot_hascontent))
print("{} Indexs exist but have no content. ".format(tot_nocontent))  # Even if the index exists, the content may be empty


# In[51]:


# Q1 = 285.75
# Mdedian = 1262.0
for i in range(0,len(df3_4["abstract"])):
    if i in Idx_rec:
        _str = df3_4["abstract"][i]
        _str = _str + df3_4["body_text"][i] + df3_4["authors"][i] + df3_4["title"][i]
        
        if len(_str)<=1262:
            df3_4["abstract"][i] = _str
        else:
            df3_4["abstract"][i] = _str[:1262]
# padding


# In[52]:


# Reprocessing Indexs have lost and exists.
sets_rec = []
Idx_rec = []
err_rec = []
for i in range(len(df3_4["abstract"])):
    try:
        sets_rec.append(split_into_word_set(df3_4["abstract"][i]))
        Idx_rec.append(i)
    except KeyError:
        sets_rec.append(set([]))
        err_rec.append(i)
print("{} Indexs have lost. ".format(len(err_rec)))
print("{} Indexs exists. ".format(len(Idx_rec)))

tot_hascontent = tot_nocontent = 0
for idx in Idx_rec:
    if len(df3_4["abstract"][idx]) == 0:
        tot_nocontent +=1
    else:
        tot_hascontent+=1
        
print()
print("{} Indexs have content. ".format(tot_hascontent))
print("{} Indexs exist but have no content. ".format(tot_nocontent))


# ### Calculation of the Jaccard matrix

# In[53]:


# Calculate the Jaccard matrix
# The following steps reflect the fact that, when we consider the complexity of finding similar vector pairs, the amount of computation required to compare everything is unmanageable even if the dataset is quite small O(n^2)
# Longer running time


# In[54]:


import time
starttime_for_Jamatrix = time.time()
K = 4  # shingle size

def build_shingles(text: str, k: int):
    shingle_set = []
    for i in range(len(text) - k+1):
        shingle_set.append(text[i:i+k])
    return set(shingle_set)


import csv
file_path = "Jaccard_matrix_K{}.csv".format(K)
Jaccard_matrix = []
if os.path.isfile(file_path):
    with open(file_path,"r") as file:
        reader = csv.reader(file)
        for row in reader:
            Jaccard_matrix.append([float(cell) for cell in row])
else:
    Jaccard_matrix = [[0 for i in range(len(df3_4["abstract"]))] for i in range(len(df3_4["abstract"]))]
    for i in range(len(df3_4["abstract"])):
        if i% 1000 ==1 : print("Running for Jaccard matrix:  ", "Line:",i,"/",len(df3_4["abstract"]), 
                                                                            " Time == ", time.time() - starttime_for_Jamatrix)
        for j in range(i+1,len(df3_4["abstract"])):
            # if i not in err_rec and j not in err_rec and len(df3_4["abstract"][i])!=0 and len(df3_4["abstract"][j])!=0:
            if i not in err_rec and j not in err_rec:
                LOWERBOUND = 286
                UPPERBOUND = 1724
                sentence0=df3_4["abstract"][i]
                sentence1=df3_4["abstract"][j]
                if len(sentence0)<=LOWERBOUND:
                    sentence0 = sentence0+df3_4['body_text'][i]
                if len(sentence0)>UPPERBOUND:
                   sentence0 = sentence0[:UPPERBOUND]
                if len(sentence1)<=LOWERBOUND:
                    sentence1 = sentence1+df3_4['body_text'][j]
                if len(sentence1)>UPPERBOUND:
                   sentence1 = sentence1[:UPPERBOUND]
                Jaccard_matrix[j][i]=Jaccard_matrix[i][j]=jaccard(sentence0,sentence1) 
    print("Total time for calculating Jaccard_matrix with len of {}".format(len(df3_4["abstract"]))," is ", time.time() - starttime_for_Jamatrix)

    # Storage Matrix
    with open (file_path,"w",newline="") as file:
        writer=csv.writer(file)
        writer.writerows(Jaccard_matrix)
    print("Jaccard matrix has been sucessfully storaged. ")


# In[55]:


# Distribution of Jaccard values
import numpy as np
import matplotlib.pyplot as plt
K=4
def __plot(mtx):
    matrix_flat = np.array(mtx).flatten()
    # matrix_flat = matrix_flat[ matrix_flat<=0.2 ]
    
    plt.hist(matrix_flat, bins=100, edgecolor='black')
    plt.title('Distribution of Jaccard_matrix K={}'.format(K))
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

__plot(Jaccard_matrix)


# ## LSH basic implementation

# In[56]:


# LSH consists of several different methods. Traditional methods (with multiple steps): shingling, MinHashing and finally banded LSH functions. 
# There is no single hashing method in LSH. In fact, they all share the same logic of "grouping similar samples by hash function", but they can be very different from each other.


# In[57]:


UPPERBOUND = 1724 # Decrease if running takes very long time e.g. 862
LOWERBOUND = 286
# Q1 286, Q3 1724
# K=4 would cost much memory, be carefull on your machine


# In[58]:


sentences = df3_4['abstract'].tolist()
for i in range(len(sentences)):
    try:
        if len(sentences[i])<=LOWERBOUND:
            sentences[i]=sentences[i]+df3_4['body_text'][i]
        if len(sentences[i])>UPPERBOUND:
            sentences[i]=sentences[i][:UPPERBOUND]
    except KeyError:
        pass
sentences[:3]


# In[59]:


def build_shingles(text: str, k: int):
    shingle_set = []
    for i in range(len(text) - k+1):
        shingle_set.append(text[i:i+k])
    return set(shingle_set)


# In[60]:


def build_vocab(singles):
    merge_single_set = set([])
    for shingle_set in singles:
        merge_single_set = merge_single_set.union(shingle_set)
    return list(merge_single_set)


# In[61]:


def one_hot(shingle_set, vocab: list):
    onehot_vec = []
    for i in range(0,len(vocab)):
        if vocab[i] in shingle_set:
            onehot_vec.append(1)
        else:
            onehot_vec.append(0)
    # len(onehot_vec ) == len(vocab)
    return onehot_vec


# In[62]:


# Generate a single alignment
import copy
def get_minhash_permutation_arr(arr0):
    from random import shuffle
    arr1 = copy.deepcopy(arr0)
    shuffle(arr1)
    return arr1
# Generate all alignments
def get_minhash_permutation_arrs(size_x,size_y,arr_seed=[]):
    arrs = []
    if len(arr_seed)!=size_x:arr_seed = [i for i in range(size_x)]
    for _ in range(size_y):
        arr_seed = copy.deepcopy(get_minhash_permutation_arr(arr_seed))
        arr_seed = copy.deepcopy(get_minhash_permutation_arr(arr_seed))
        arrs.append(arr_seed)
    return arrs


# In[63]:


# Generate 1 signature
def get_signature(vector, permutation_arrs):
    signature = []
    for permutation_arr in permutation_arrs:
        flag = False
        for i in range(0,len(permutation_arr)):
            if vector[permutation_arr[i]]==1:
                signature.append(i)
                flag = True
                break
        if not flag: signature.append(len(signature))
    return signature


# In[64]:


K = 4  # shingle size

# build shingles
shingles = []
for sentence in sentences:
    shingles.append(build_shingles(sentence, K))

# build vocab
vocab = build_vocab(shingles)
# if len(vocab)>150000: vocab=vocab[:150000] #truncate

print(len(vocab))


# ### Shingling

# In[65]:


# K - len(vocab)
leng_list = []
costT_list = []
K_list = []


# In[66]:


# build shingles
shingles = []
for sentence in sentences:
    shingles.append(build_shingles(sentence, K))

# build vocab
vocab = build_vocab(shingles)
# if len(vocab)>150000: vocab=vocab[:150000] #truncate


# In[67]:


print(len(vocab))
print(len(shingles))
print(len(shingles[0]))


# ### building onehot - first slice
# if use total 8028 length at a time, Memery Error
# so divided into 2 slices averagably, and do two times

# In[68]:


# one-hot encode our shingles
shingles_1hot = []
__cnt = 0
for i in range(0,int(len(shingles)/2)):
    shingle_set = shingles[i]
    if __cnt % 100 == 0: print("This is ", __cnt)
    __cnt+=1
    shingles_1hot.append(one_hot(shingle_set, vocab))


# In[69]:


print(len(shingles_1hot))
print(len(shingles_1hot[0]))
print(len(vocab))
# Too long a vocab can lead to a memory explosion, especially if you're using shingling.
# K = 8 , len(vocab) == 2195986 It's too long.
# K = 4


# In[70]:


sum(shingles_1hot[0])  # confirm we have 1s


# ### Minhash - first time

# In[71]:


permutation_arrs = get_minhash_permutation_arrs(size_x=len(vocab), size_y=100)

signatures = []
cnt = 0
for vector in shingles_1hot:
    cnt = cnt + 1
    if cnt % 100 ==0: print(len(signatures),"/",len(shingles_1hot))
    signatures.append(get_signature(vector, permutation_arrs))

# merge signatures into single array
# signatures = np.stack(signatures)
# signatures.shape


# In[72]:


print(np.array(signatures[:10][:10]))


# ### building onehot - second slice
# if use total 8028 length at a time, Memery Error
# so divided into 2 slices averagably, and do two times

# In[73]:


# one-hot encode our shingles - carry on
shingles_1hot = []
for i in range(int(len(shingles)/2),len(shingles)):
    shingle_set = shingles[i]
    if __cnt % 100 == 0: print("This is ", __cnt)
    __cnt+=1
    shingles_1hot.append(one_hot(shingle_set, vocab))


# ### Minhash - second time

# In[74]:


for vector in shingles_1hot:
    cnt = cnt + 1
    if cnt % 100 ==0: print(len(signatures),"/",len(shingles_1hot)*2)
    signatures.append(get_signature(vector, permutation_arrs))

# merge signatures into single array
signatures = np.stack(signatures)
signatures.shape


# In[75]:


# calcel shingles_1hot
shingles_1hot = None


# ### LSH 

# In[76]:


# LSH : Finally, we move onto the LSH process. We will use a class here:
from itertools import combinations

class LSH:
    buckets = []
    counter = 0
    def __init__(self, b):
        self.b = b
        for i in range(b):
            self.buckets.append({})

    def make_subvecs(self, signature):
        l = len(signature)
        assert l % self.b == 0
        r = int(l / self.b)
        # break signature into subvectors
        subvecs = []
        for i in range(0, l, r):
            subvecs.append(signature[i:i+r])
        return np.stack(subvecs)
    
    def add_hash(self, signature):
        subvecs = self.make_subvecs(signature).astype(str)
        for i, subvec in enumerate(subvecs):
            subvec = ','.join(subvec)
            if subvec not in self.buckets[i].keys():
                self.buckets[i][subvec] = []
            self.buckets[i][subvec].append(self.counter)
        self.counter += 1

    def check_candidates(self):
        candidates = []
        for bucket_band in self.buckets:
            keys = bucket_band.keys()
            for bucket in keys:
                hits = bucket_band[bucket]
                if len(hits) > 1:
                    candidates.extend(combinations(hits, 2))
        return set(candidates)


# In[77]:


band = 20
lsh = LSH(band)

for signature in signatures:
    lsh.add_hash(signature)


# In[78]:


lsh.buckets


# In[79]:


candidate_pairs = lsh.check_candidates()
len(candidate_pairs)


# In[80]:


list(candidate_pairs)[:5]


# ###  P-s plot result - Optimizing the Bands

# In[81]:


## Optimizing the Bands

# Now let's visualize the actual cosine similarity of our signature vectors against whether we identified the signatures as candidate pairs or not.

# (we will also calculate Jaccard but it's less useful here, try both!)


# In[82]:


import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics.pairwise import cosine_similarity


def jaccard_for_set(set1,set2):
    # Calculate the length of the intersection intersection and the union union of two sets. 
    # Finally we use the ratio of intersection to union to compute the Jaccard coefficient
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union==0:return 1 # Two empty sets
    jaccard_coefficient = intersection / union
    return jaccard_coefficient

pairs = pd.DataFrame({
    'x': [],
    'y': [],
    'jaccard': [],
    'cosine': [],
    'candidate': []
})

data_len = len(signatures)
chosen = set()
# take random sample of pairs
sample_size = 50_000
for _ in range(sample_size):
    x, y = np.random.choice(data_len, 2)
    if x == y or (x, y) in chosen: continue
    chosen.add((x, y))
    vector_x = signatures[x]
    vector_y = signatures[y]
    candidate = 1 if (x, y) in candidate_pairs else 0
    cosine = cosine_similarity([vector_x], [vector_y])[0][0]
    pairs = pd.concat([pairs,pd.DataFrame([{
        'x': x,
        'y': y,
        'jaccard': jaccard_for_set(set(vector_x), set(vector_y)),
        'cosine': cosine,
        'candidate': candidate
    }])], ignore_index=True)


# add a normalized cosine column for better alignment
cos_min = pairs['cosine'].min()
cos_max = pairs['cosine'].max()
pairs['cosine_norm'] = (pairs['cosine'] - cos_min) / (cos_max - cos_min)


# In[83]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(data=pairs, x='cosine', y='candidate', alpha=0.5)


# In[84]:


def probability(s, r, b):
    # s: similarity
    # r: rows (per band)
    # b: number of bands
    return 1 - (1 - s**r)**b

def normalize(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)


# In[85]:


b = 20
r = int(100 / b)
s_scores = np.arange(0.01, 1, 0.01)
P_scores = [probability(s, r, b) for s in s_scores]

sns.lineplot(x=s_scores, y=P_scores)
sns.scatterplot(data=pairs, x='cosine', y='candidate', alpha=0.1, color='k')


# In[86]:


b = 25
r = int(100 / b)
s_scores = np.arange(0.01, 1, 0.01)
P_scores = [probability(s, r, b) for s in s_scores]

sns.lineplot(x=s_scores, y=P_scores)
sns.scatterplot(data=pairs, x='cosine_norm', y='candidate', alpha=0.1, color='k')


# In[87]:


# Grid Search b
probs = pd.DataFrame({
    'P': [],
    's': [],
    'b': []
})

for b in [100, 50, 25, 20, 10, 5, 2]:
    r = int(100 / b)
    s_scores = np.arange(0.01, 1, 0.01)
    P_scores = [probability(s, r, b) for s in s_scores]
    probs = pd.concat([probs, pd.DataFrame({
        'P': P_scores,
        's': s_scores,
        'band': [str(b)]*len(s_scores)
    })], ignore_index =True)

sns.lineplot(data=probs, x='s', y='P', hue='band')
print("We will choose the green line (band = 25)")


# ## Experiments using implemented models

# In[88]:


get_ipython().system('pip install datasketch')


# In[89]:


# pick 'abstract' column as text

from datasketch import MinHash, MinHashLSH
print(df3_4["abstract"].head(10))


# In[90]:


MinHash_list = [MinHash(num_perm=128) for i in range(len(df3_4["abstract"]))]  # MinHash for all
for i in range(len(Idx_rec)):
    idx = Idx_rec[i]
    for d in sets_rec[idx]:
        MinHash_list[idx].update(d.encode('utf-8'))


# In[91]:


# Create LSH index
lsh = MinHashLSH(threshold=0.5, num_perm=128)  # hyperparameterisation

query = [1] # take paper 1 as example
# All papaer have no content(no abstract)

for i in range(len(Idx_rec)):
    idx = Idx_rec[i]
    if idx not in query:
        lsh.insert(str(idx),MinHash_list[idx])

result = lsh.query(MinHash_list[1])
print("LEN of result: ", len(result))


# In[92]:


# Create LSH index
# e.g.
lsh = MinHashLSH(threshold=0.5, num_perm=128)

query = [3] # take paper 0 as example
# All papaer have no content(no abstract)

for i in range(len(Idx_rec)):
    idx = Idx_rec[i]
    if idx not in query:
        lsh.insert(str(idx),MinHash_list[idx])

result = lsh.query(MinHash_list[3])
print("LEN of result: ", len(result))
print("Approximate neighbours with Jaccard similarity > 0.5", result)   # The similarity of the vast majority of articles is not high.


# In[93]:


# The Jaccard matrix has been preprocessed.

def verify_LSH_candidates(picked_columnk,_result,threshold):
    if picked_columnk in err_rec: return 0,0,0,0  # TP,FP,TN,FN
    
    TP = FP = 0
    for candidate in _result:
        if int(candidate) in err_rec:continue
        if Jaccard_matrix[picked_columnk][int(candidate)] >= float(threshold) or (len(df3_4["abstract"][picked_columnk])==0 and len(df3_4["abstract"][int(candidate)])==0): # 0集特判
            TP = TP + 1
        else:
            FP = FP + 1    
    TN = FN = 0
    for idx in range(len(df3_4["abstract"])):
        if (str(idx) not in _result) and (idx not in err_rec):
            # idx is predicted Negetive
            if Jaccard_matrix[picked_columnk][int(candidate)] < float(threshold) and not (len(df3_4["abstract"][picked_columnk])==0 and len(df3_4["abstract"][int(candidate)])==0): # 0集特判
                TN = TN + 1
        else:
            FN = FN + 1
    return TP, FP, TN, FN
    
TP, FP, TN, FN = verify_LSH_candidates(picked_columnk=3, _result=result, threshold=0.5)
print(TP/(TP+FP))

print("True Positive candidate: ",TP, "False Negetive candidate: ", FP) # e.g.


# In[94]:


import math
segment_leng = int(math.sqrt(len(df3_4["abstract"]))) # Root time and space complexity


# In[95]:


# Explore MinHashing, threshold, TP/(TP+FP) for subsequent plotting of Accuracy's percentage stacks

# Preprocessing lsh prefixes suffixes Optimise time, reduce computation (but preprocessing is also slow)
# Preprocessing lsh prefixes and suffixes was still slow and prone to memory blowups, and eventually chunking was adopted
# Given hyperparameters threshold, num_perm
import sys
def get_lsh_maxtrix(threshold,num_perm):
    lsh_segment = {}
#    segment_leng = int(math.sqrt(len(df3_4["abstract"]))) # Root time and space complexity
    segment_size = []
    for i in range(0,len(df3_4["abstract"]),segment_leng):
        end_i = i
        lsh_example = MinHashLSH(threshold=threshold, num_perm=num_perm)
        while end_i < i + segment_leng and end_i < len(df3_4["abstract"]):
            lsh_example.insert(str(end_i),MinHash_list[end_i])
            end_i+=1
        lsh_segment[ str(i)+"-"+str(end_i-1) ] = copy.deepcopy(lsh_example)
        segment_size.append([i,sys.getsizeof(lsh_segment)])
        if i%100 == 0: 
            print("threshold:{} num_perm:{}".format(threshold, num_perm), "Segment interval:", i, "---", end_i-1, "Size of lsh dict record: ", sys.getsizeof(lsh_segment), "bytes")
    return lsh_segment

# get_lsh_maxtrix(threshold = 0.5, num_perm = 128)
# This code can be used to increase the average speed of a query with the hyperparameters threshold, num_perm unchanged.


# In[96]:


# Watch out for memory crashes


# ### Try different thresholds

# In[97]:


threshold_List = [0.05, 0.07, 0.10, 0.13, 0.15, 0.17]
# decimal = 2
Accuracy_list = []
idx_list = []
x_cnt = 0
# util array
totTP_rec = []
totTN_rec = []
totFP_rec = []
totFN_rec = []
start_time = time.time()
for fractional in threshold_List:    # Test the confidence level of threshold
    threshold = float(fractional)
    print("Now threshold is {}".format(threshold))
    start_t = time.time()
    lsh_segment = get_lsh_maxtrix(threshold=threshold, num_perm=128)  # Long running ......
    print("get lsh segments time(Complexity{} * {}): ".format(segment_leng, segment_leng), time.time() - start_t)
    
    # Calculate the TP, FP, TN, FN values for all feasible papers (with abstracts).
    # util array
    TP_rec = []
    TN_rec = []
    FP_rec = []
    FN_rec = []
    totTP = totFP = totTN = totFN =0

    # Sampling, 100 samples (in 8000 samples), significant efficiency gains
    import random
    all_values = list(range(0,len(df3_4["abstract"])))
    selected_values = random.sample(all_values, 20)

    __cnt = 0
    for idx in selected_values:   
        __cnt += 1
        if idx not in Idx_rec: continue
        # Create new LSH index for Accuracy plot
        lsh_context = MinHashLSH(threshold=threshold, num_perm=128)
        for i in range(0,len(df3_4["abstract"]),segment_leng):
            if i+segment_leng<len(df3_4["abstract"]):
                end_i = i+segment_leng-1
            else:
               end_i = len(df3_4["abstract"])-1
            if idx>=i and idx<=end_i:
                lsh_example = MinHashLSH(threshold=threshold, num_perm=128)
                for j in range(i,end_i+1):
                    if j in Idx_rec: lsh_example.insert(str(j),MinHash_list[j])
                lsh_context.merge(lsh_example)
            else:
                lsh_context.merge(lsh_segment[str(i)+"-"+str(end_i)])
                # prefix-suffix splice
        
        result = lsh_context.query(MinHash_list[idx])  # Find LSH candidates for paper idx
        
        print("        Index=",idx,len(df3_4["abstract"][idx]), "[{}/{}]".format(__cnt,len(selected_values)))
        TP, FP, TN, FN = verify_LSH_candidates(picked_columnk=idx, _result=result, threshold=threshold)
        print("        Now threshold is {}".format(threshold), TP+TN, FP+FN, TP, "Accuracy = {}".format((TP+TN)/(TP+TN+FP+FN)), 
              "Cumulative time：", time.time()-start_time)
        totTP += TP
        totFP += FP
        totTN += TN
        totFN += FN
        
        TP_rec.append(TP)
        TN_rec.append(TN)
        FP_rec.append(FP)
        FN_rec.append(FN)
    # util array
    totTP_rec.append(totTP)
    totTN_rec.append(totTN)
    totFP_rec.append(totFP)
    totFN_rec.append(totFN)
    Accuracy_list.append((totTP+totTN)/(totTP+totTN+totFP+totFN))
    idx_list.append(fractional)
    x_cnt+=1


# In[98]:


# Plotting stacked plots threshold - Accuracy TP-TN-FP-FN Percentage stacked plots
# To plot a TP-TN-FP-FN percentage stacked plot, you can use the matplotlib library for visualisation. Here is the code to plot a percentage stacked plot (Accuracy)
import matplotlib.pyplot as plt
categories = []
for _f in idx_list:
    categories.append(str(_f))

# Calculate the percentage of data for each series
total = [sum(x) for x in zip(totTP_rec, totTN_rec, totFP_rec,totFN_rec)]
totTP_rec_percent = [x / t * 100 for x, t in zip(totTP_rec, total)]
totTN_rec_percent = [x / t * 100 for x, t in zip(totTN_rec, total)]
totFP_rec_percent = [x / t * 100 for x, t in zip(totFP_rec, total)]
totFN_rec_percent = [x / t * 100 for x, t in zip(totFN_rec, total)]

# threshold = [0.05, 0.07, 0.10, 0.13, 0.15, 0.17]
print("[TP , TN , FP , FN], num = {}".format(len(totTP_rec)))
for i in range(len(totTP_rec)):
    print("[ ",totTP_rec_percent[i],",",totTN_rec_percent[i],",",totFP_rec_percent[i],",",totFN_rec_percent[i]," ]")

# Plotting percentage stacks
Q1 = [0 for _ in range(len(totTP_rec_percent))]
Q2 = [totTP_rec_percent[i] for i in range(len(totTP_rec_percent))]
Q3 = [totTP_rec_percent[i] + totTN_rec_percent[i]  for i in range(len(totTP_rec_percent))]
Q4 = [totTP_rec_percent[i] + totTN_rec_percent[i]  + totFP_rec_percent[i] for i in range(len(totTP_rec_percent))]

plt.bar(categories, totTP_rec_percent, bottom=Q1, label='TP')
plt.bar(categories, totTN_rec_percent, bottom=Q2, label='TN')
plt.bar(categories, totFP_rec_percent, bottom=Q3, label='FP')
plt.bar(categories, totFN_rec_percent, bottom=Q4, label='FN')

# Setting the title and legend
plt.title('Percentage Stacked Bar Chart of different threshold')
plt.legend()

# Display Graphics
plt.show()

print("Accuracy_list")
print(idx_list)
print(Accuracy_list)


# ### Optimising similarity calculations using different families and parameters of LSH

# In[99]:


# Goal Less time + higher accuracy


# In[100]:


# result：Getting the best num_perm and threshold using grid search
# Time calculations must be significantly optimised，Need to use lsh prefix suffix merge optimisation
# Grid Search
permutation_parameters = [8,32,128,512,2048]
threshold_parameters = [0.03 , 0.05 , 0.07 , 0.10 , 0.13, 0.15, 0.17]

P_x = []
T_y = []
Accuracy_z = []

import math

for perm1 in permutation_parameters:
    for thr1 in threshold_parameters:
        num_perm = perm1
        threshold = float(thr1)
        print("Now threshold is {} ||| Perm number is {}".format(threshold,num_perm))
        start_t = time.time()
        lsh_segment = get_lsh_maxtrix(threshold=threshold, num_perm=128)  # Long running ......
        print("get lsh segments time ( Complexity {} * {} ): ".format(segment_leng, segment_leng), time.time() - start_t)

        totTP = totFP = totTN = totFN =0
        
        import random
        all_values = list(range(0,len(df3_4["abstract"])))
        selected_values = random.sample(all_values, 100)
        
        for idx in selected_values:
            if idx not in Idx_rec: continue
            # Create new LSH index for Accuracy plot
            lsh_context = MinHashLSH(threshold=threshold, num_perm=128)
            for i in range(0,len(df3_4["abstract"]),segment_leng):
                if i+segment_leng<len(df3_4["abstract"]):
                    end_i = i+segment_leng-1
                else:
                   end_i = len(df3_4["abstract"])-1
                if idx>=i and idx<=end_i:
                    lsh_example = MinHashLSH(threshold=threshold, num_perm=128)
                    for j in range(i,end_i+1):
                        if j in Idx_rec: lsh_example.insert(str(j),MinHash_list[j])
                    lsh_context.merge(lsh_example)
                else:
                    lsh_context.merge(lsh_segment[str(i)+"-"+str(end_i)])
            
            result = lsh_context.query(MinHash_list[idx])
            totTP += TP
            totFP += FP
            totTN += TN
            totFN += FN
            
        P_x.append(int(math.log(perm1,2)))
        T_y.append(threshold)
        Accuracy_z.append( (totTP+totTN) / (totTP+totTN+totFP+totFN) )
        
        end_time = time.time()
        print("Time for one round：", end_time-start_time)
    
print("【Cumulative time to test the confidence level of threshold , perm】 :", time.time()-start_time)


# Accuracy


# In[101]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

triplet1 = tuple(P_x)
triplet2 = tuple(T_y)
triplet3 = tuple(Accuracy_z)

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111,projection="3d")

scatter = ax.scatter(np.array(triplet1), np.array(triplet2), np.array(triplet3), c=np.array(triplet3), cmap = "plasma", s=100)
fig.colorbar(scatter)

ax.set_zlim([np.min(triplet3),np.max(triplet3)])

ax.set_xlabel("Log2 - permutation number")
ax.set_ylabel("Threshold")
ax.set_zlabel("Accuracy")
ax.set_title("3D-perm-threshold-Accuracy Distribution(K=4)")

plt.show()


# ## Exploring the use of word embeddings

# In[102]:


# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
# documents = df3_4["abstract"].tolist()
documents = sentences
print("Number of texts：",len(documents))
# Creating a TF-IDF vectoriser
vectorizer = TfidfVectorizer()
# TF-IDF conversion of text data
tfidf_matrix = vectorizer.fit_transform(documents)

# Printable Vocabulary List
try:
    print("Vocabulary: ", vectorizer.get_feature_names(), len(vectorizer.get_feature_names()))
except AttributeError:
    print("Vocabulary: ", vectorizer.get_feature_names_out(), len(vectorizer.get_feature_names_out()))


# In[103]:


# Print the TF-IDF weight matrix
print("TF-IDF Matrix: ")
print(tfidf_matrix.toarray())


# ### Account for semantic similarity

# In[104]:


import numpy as np
def cos_sim(vec1,vec2):
    dot_product=np.dot(vec1,vec2)
    norm_vec1=np.linalg.norm(vec1)
    norm_vec2=np.linalg.norm(vec2)
    _sim=dot_product / (norm_vec1 * norm_vec2)
    return _sim

cos_values = []
# Tfidf_matrix = tfidf_matrix.toarray() Memory crush
for i in range(tfidf_matrix.shape[0]):
    for j in range(i+1,tfidf_matrix.shape[0]):
        cos_values.append(cos_sim(tfidf_matrix[i].toarray()[0],tfidf_matrix[j].toarray()[0]))
import sys
print(sys.getsizeof(cos_values))
print(type(cos_values))

# Plotting the distribution of values
import seaborn as sns
sns.kdeplot(cos_values)

tfidf_matrix = None
Tfidf_matrix = None
cos_values = None


# In[105]:


# Similarity Analysis cancellation variable:

df3_4=Jaccard_matrix=None
sets_rec=Idx_rec=err_rec=None
sentences=shingles=vocab=None
leng_list=costT_list=K_list=None
shingles_1hot=permutation_arrs=lsh=candidate_pairs=None
pairs=MinHash_list=result=None


# In[ ]:





# # Clustering Analysis

# In[106]:


# Vectorization using TF-IDF
# we will be using tf-idf. This will convert our string formatted data into a measure of how important each word is to the instance out of the literature as a whole.


# ## Feature engineering of text

# In[107]:


df3_5 = df.copy(deep=True)
nlp = spacy.load('en_core_web_sm')

# defined custom stop words list
my_stop_words = ['doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure',
    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.',
    'al.', 'Elsevier', 'PMC', 'CZI', ' ', '\n', '  ', '+', ' \n', '  \n']
for stopword in my_stop_words:
    nlp.vocab[stopword].is_stop = True


tqdm.pandas()
df3_5['cleaned_abstract'] = df3_5['abstract'].progress_apply(clean_text)
df3_5['processed_abstract'] = df3_5['cleaned_abstract'].progress_apply(spacy_process_text)

df3_5['cleaned_text'] = df3_5['body_text'].progress_apply(clean_text)
df3_5['processed_text'] = df3_5['cleaned_text'].progress_apply(spacy_process_text)

print(df3_5[['processed_abstract']])
print(df3_5[['processed_text']])


# In[108]:


from sklearn.feature_extraction.text import TfidfVectorizer
def vectorize(text, maxx_features):

    vectorizer = TfidfVectorizer(max_features=maxx_features)
    X = vectorizer.fit_transform(text)
    return X


text = df3_5['processed_text'].values
max_features = 2**12

X = vectorize(text, max_features)
print(X) #Each line represents a word in a document and its TF-IDF value
#formatted as (document index, word index) TF-IDF value.


# In[109]:


abstract = df3_5['processed_abstract'].values

Y = vectorize(abstract, max_features)
print(Y)
print(Y.shape)


# ## Topic clustering: KMeans Algorithm

# In[110]:


from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans

# PCA
pca = PCA(n_components=0.95, random_state=42)
X_reduced= pca.fit_transform(X.toarray())


# KMeans clustering
k = 10
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans_lables_text = kmeans.fit_predict(X_reduced)
df3_5['kmeans_cluster_text'] = kmeans_lables_text
df3_5.head()


# In[111]:


Y_reduced= pca.fit_transform(Y.toarray())

kmeans_lables_abstract = kmeans.fit_predict(Y_reduced)
df3_5['kmeans_cluster_abstract'] = kmeans_lables_abstract
df3_5.head()


# ## Topic clustering: DBSCAN Algorithm

# In[112]:


from sklearn.neighbors import NearestNeighbors


# Calculate the distance from each point to its 5th nearest neighbor
k1 = 4429
neighbors = NearestNeighbors(n_neighbors=k1)
neighbors_fit = neighbors.fit(X_reduced)
distances, indices = neighbors_fit.kneighbors(X_reduced)

# Sorting distances
distances = np.sort(distances[:, k1-1], axis=0)

# Plotting the K-distance
plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.xlabel("Sample Index")
plt.ylabel(f"Distance to {k1}th Nearest Neighbor")
plt.title("K-Distance Graph")
plt.show()


# In[113]:


from sklearn.cluster import DBSCAN
# Applying DBSCAN Clustering
dbscan_text = DBSCAN(eps=1.36, min_samples=2216)
dbscan_labels_text = dbscan_text.fit_predict(X_reduced)
df3_5['dbscan_cluster_text'] = dbscan_labels_text
df3_5.head()


# In[114]:


from sklearn.cluster import AgglomerativeClustering
# Apply hierarchical clustering
agglo = AgglomerativeClustering(n_clusters=10)
agglo_labels_text = agglo.fit_predict(X_reduced)
df3_5['agglo_cluster_text'] = agglo_labels_text
df3_5.head()


# In[115]:


agglo_labels_abstract = agglo.fit_predict(Y_reduced)
df3_5['agglo_cluster_abstract'] = agglo_labels_abstract
df3_5.head()


# ## TSNE visualization

# In[116]:


from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns


tsne = TSNE(verbose=1, perplexity=50)  # Changed perplexity from 100 to 50 per FAQ
X_embedded = tsne.fit_transform(X.toarray())


# sns settings
sns.set(rc={'figure.figsize':(15,15)})

# colors
palette = sns.color_palette("bright", 1)

# plot
sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], palette=palette)

plt.title('t-SNE with no Labels_body_text')
plt.savefig("t-sne_covid19.png")
plt.show()


# In[117]:


Y_embedded= tsne.fit_transform(Y.toarray())

sns.scatterplot(x=Y_embedded[:,0], y=Y_embedded[:,1], palette=palette)#x=Y_embedded[:, 0] 和 y=Y_embedded[:, 1] 分别指定了降维后数据的第一个和第二个主成分作为 x 轴和 y 轴
plt.title('t-SNE with no Labels_abstract')
plt.savefig("t-sne_covid19_abstract.png")
plt.show()


# In[118]:


# sns settings
sns.set(rc={'figure.figsize':(13,9)})

# colors
palette = sns.hls_palette(20, l=.4, s=.9)

# plot
sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=kmeans_lables_text, legend='full', palette=palette)
plt.title('t-SNE with Kmeans Labels')
plt.savefig("k_means_improved_cluster_tsne.png")
plt.show()


# In[119]:


sns.scatterplot(x=Y_embedded[:,0], y=Y_embedded[:,1], hue=kmeans_lables_abstract, legend='full', palette=palette)
plt.title('t-SNE with abstract_Kmeans Labels')
plt.savefig("k_means_abstract_cluster_tsne.png")
plt.show()


# In[120]:


from collections import Counter
k=10
for i in range(k):
    cluster_data = df3_5[df3_5['kmeans_cluster_text'] == i]['processed_text']
    all_words = ' '.join(cluster_data).split()
    word_counts = Counter(all_words)
    common_words = word_counts.most_common(10)  # Take the top 10 common words
    print(f"Cluster {i} common words:")
    for word, count in common_words:
        print(f" - {word}: {count}")


# In[121]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# TF-IDF vectorization using TfidfVectorizer
def tfidf_vectorize(text, max_features):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(text)
    return X, vectorizer

# Perform thematic analysis on each cluster
def topic_modeling_per_cluster(df, n_clusters, n_topics, max_features=2**12, n_top_words=10):
    # Get text data
    text = df3_5['processed_text'].values

    # TF-IDF vectorize all text
    X, vectorizer = tfidf_vectorize(text, max_features)

    # Create an LDA model for each cluster
    lda_models = []
    for cluster in range(n_clusters):
        cluster_text = df3_5[df3_5['kmeans_cluster_text'] == cluster]['processed_text'].values
        if len(cluster_text) > 0:
            X_cluster, _ = tfidf_vectorize(cluster_text, max_features)
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(X_cluster)
            lda_models.append(lda)
        else:
            lda_models.append(None)

    # Print words for each topic
    for cluster, lda in enumerate(lda_models):
        if lda is not None:
            print(f"Cluster {cluster}:")
            feature_names = vectorizer.get_feature_names_out()
            for topic_idx, topic in enumerate(lda.components_):
                print(f"  Topic #{topic_idx}:")
                print("  ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        else:
            print(f"Cluster {cluster}: No data")

#Print words for each topic
n_clusters = 10  # Number of clusters
n_topics = 5  # Number of topics per cluster
max_features = 2**12  # Maximum number of features

# Calling functions for topic analysis
topic_modeling_per_cluster(df3_5, n_clusters, n_topics, max_features)


# In[122]:


# plot
palette[1] = (0, 0, 0)
sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], legend='full', palette=palette)
plt.title('t-SNE with DBSCAN Labels')
plt.savefig("DBSCAN_improved_cluster_tsne.png")
plt.show()


# In[123]:


sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=agglo_labels_text , legend='full', palette=palette)
plt.title('t-SNE with AgglomerativeClustering Labels')
plt.savefig("agglo_improved_cluster_tsne.png")
plt.show()


# ## Feature engineering of publish time

# In[124]:


from sklearn.preprocessing import StandardScaler
from datetime import datetime
new_df= pd.DataFrame()
new_df['publish_time'] = pd.to_datetime(df3_5['publish_time'], errors='coerce')
new_df["title"] = df3_5["title"]

# Extract year, month, and day features
new_df['year'] = new_df['publish_time'].dt.year
new_df['month'] = new_df['publish_time'].dt.month
new_df['day'] = new_df['publish_time'].dt.day

print(new_df.shape)
print(new_df['year'].shape)
print(new_df['month'].shape)
print(new_df['day'].shape)
new_df.info()

# Check for NaN values
print(new_df[['year', 'month', 'day']].isna().sum())

# Drop rows with NaN values
df_cleaned = new_df.dropna(subset=['year', 'month', 'day'])

# Check if data still has NaN values
print(df_cleaned[['year', 'month', 'day']].isna().sum())
print(df_cleaned[['publish_time', 'year', 'month', 'day']].head())

# Standardize year, month, and day features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_cleaned[['year', 'month', 'day']])
scaled_features_df = pd.DataFrame(scaled_features, columns=['year_scaled', 'month_scaled', 'day_scaled'])

# ombine standardized features back to the original DataFrame
df_cleaned = df_cleaned.reset_index(drop=True)
df_cleaned = pd.concat([df_cleaned, scaled_features_df], axis=1)
df_cleaned.info()


# ## Publish time clustering

# In[125]:


from sklearn.neighbors import NearestNeighbors

# Calculate the distance from each point to its 5th nearest neighbor
k2 = 5
neighbors = NearestNeighbors(n_neighbors=k2)
neighbors_fit = neighbors.fit(df_cleaned[['year_scaled', 'month_scaled', 'day_scaled']])
distances, indices = neighbors_fit.kneighbors(df_cleaned[['year_scaled', 'month_scaled', 'day_scaled']])

# Sorting distances
distances = np.sort(distances[:, k2-1], axis=0)

# Plotting the K-distance
plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.xlabel("Sample Index")
plt.ylabel(f"Distance to {k2}th Nearest Neighbor")
plt.title("K-Distance Graph")
plt.show()


# In[126]:


from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.6, min_samples=6)
df_cleaned['dbscan_cluster'] = dbscan.fit_predict(df_cleaned[['year_scaled', 'month_scaled', 'day_scaled']])
df_cleaned.info()

df_cleaned.head()


# In[127]:


# visualize DBSCAN clustering result
sns.set(rc={'figure.figsize':(10, 6)})
sns.scatterplot(x='month', y='year', hue='dbscan_cluster', data=df_cleaned, palette='tab10', s=100)
plt.title('DBSCAN Clustering based on Publish Date')
plt.xlabel('Month')
plt.ylabel('Year')
plt.legend(title='Cluster')
plt.show()


# In[128]:


#3D scatter
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(df_cleaned['year'], df_cleaned['month'], df_cleaned['day'], c=df_cleaned['dbscan_cluster'], cmap='tab10', s=50)
plt.colorbar(sc, ax=ax)
ax.set_xlabel('Year')
ax.set_ylabel('Month')
ax.set_zlabel('Day')
plt.title('3D Scatter Plot of DBSCAN Clustering based on publish date')
plt.show()


# In[129]:


from collections import Counter
cluster_counts = df_cleaned['dbscan_cluster'].value_counts()
print(cluster_counts)


# In[130]:


import pandas as pd

# Create a new column 'year_month' in the format 'YYYY-MM'
df_cleaned['year_month'] = df_cleaned['year'].astype(str) + '-' + df_cleaned['month'].astype(str).str.zfill(2)

# Count the frequency of each 'year_month' occurrence
year_month_freq = df_cleaned['year_month'].value_counts().sort_index()
print(year_month_freq)


# In[131]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# Ensure that the year_month column is of string type
df_cleaned['year_month'] = df_cleaned['year_month'].astype(str)

# Handling Floating Point Numbers in the year_month Column
df_cleaned['year'] = df_cleaned['year_month'].apply(lambda x: int(float(x.split('-')[0])))
df_cleaned['month'] = df_cleaned['year_month'].apply(lambda x: int(float(x.split('-')[1])))

# Count the frequency of each (year, month)
freq_table = df_cleaned.groupby(['year', 'month']).size().reset_index(name='frequency')

# Creating 3D charts
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# Extract data
x = freq_table['year']
y = freq_table['month']
z = freq_table['frequency']

# Generate color maps
colors = cm.viridis(z / max(z))

# Creating 3D bar charts
ax.bar3d(x, y, np.zeros(len(z)), 1, 1, z, shade=True, color=colors)

# Setting axis labels
ax.set_xlabel('Year')
ax.set_ylabel('Month')
ax.set_zlabel('Frequency')

# Setting the title
plt.title('3D Bar Plot of Year-Month Frequency')

plt.show()


# In[132]:


for cluster in df_cleaned['dbscan_cluster'].unique():
    cluster_data = df_cleaned[df_cleaned['dbscan_cluster'] == cluster]
    print(f"Cluster {cluster} Titles:")
    print(cluster_data)
    print(cluster_data['title'].values)
    print("\n")


# In[133]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction import text

# Get default English stop words and add custom stop words
default_stopwords = text.ENGLISH_STOP_WORDS
custom_stopwords = list(default_stopwords) + ['br']  # Add your customized stop words here

for cluster in df_cleaned['dbscan_cluster'].unique():
    cluster_data = df_cleaned[df_cleaned['dbscan_cluster'] == cluster]

    # Initialize TfidfVectorizer with an expanded list of deactivated words
    tfidf = TfidfVectorizer(stop_words=custom_stopwords)
    tfidf_matrix = tfidf.fit_transform(cluster_data['title'])

    # Training LDA Models
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(tfidf_matrix)

    print(f"Topics in Cluster {cluster}:")
    for idx, topic in enumerate(lda.components_):
        print(f"Topic {idx}:")
        print([tfidf.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
    print("\n")


# In[134]:


# Setting Hierarchical Clustering Parameters
agg_clustering = AgglomerativeClustering(n_clusters=10)

# Clustering of normalized temporal features
df_cleaned['hierarchical_cluster'] = agg_clustering.fit_predict(df_cleaned[['year_scaled', 'month_scaled', 'day_scaled']])
print(df_cleaned[['publish_time', 'hierarchical_cluster']])
df_cleaned.info()


# In[135]:


sns.set(rc={'figure.figsize':(10, 6)})
sns.scatterplot(x='month', y='year', hue='hierarchical_cluster', data=df_cleaned, palette='tab10', s=100)
plt.title('Hierarchical Clustering based on Publish Date')
plt.xlabel('Month')
plt.ylabel('Year')
plt.legend(title='Cluster')
plt.show()


# In[136]:


df_cleaned


# In[137]:


# Get Cluster Centers
cluster_centers = df_cleaned[:10]
# Create a DataFrame to hold the center point
cluster_centers_df = pd.DataFrame(cluster_centers, columns=['year_scaled', 'month_scaled', 'day_scaled'])
print(cluster_centers_df)


# In[138]:


# Reducing the center point to the original scale
cluster_centers_original = scaler.inverse_transform(cluster_centers_df)
print(cluster_centers_original)
cluster_centers_original_df = pd.DataFrame(cluster_centers_original, columns=['year', 'month', 'day'])
print(cluster_centers_original_df)


# In[139]:


# Visualizing KMeans Clustering Results and Cluster Centroids
sns.set(rc={'figure.figsize':(10, 6)}) # K-means_cluster
sns.scatterplot(x='month', y='year', hue='hierarchical_cluster', data=df_cleaned, palette='tab10', s=100)
plt.scatter(cluster_centers_original_df['month'], cluster_centers_original_df['year'], s=300, c='red', marker='X', label='Centroids')
plt.title('KMeans Clustering and Cluster Centers based on Publish Date')
plt.xlabel('Month')
plt.ylabel('Year')
plt.legend(title='Cluster')
plt.show()


# In[140]:


# Counting the amount of data per cluster
#cluster_counts = df_cleaned['K-means_cluster'].value_counts()
#print(cluster_counts)
print("K_means_cluster")
print("0    1198")
print("7    1196")
print("3    1137")
print("5    1016")
print("1     964")
print("9     882")
print("8     876")
print("4     293")
print("6     266")
print("2     139")
print("Name: count, dtype: int64")


# # End

# In[141]:


import time
print("Time usage",time.time()-STARTTIME)

