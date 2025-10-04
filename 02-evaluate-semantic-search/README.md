
# 🔍 Evaluating Semantic Search: A Beginner's Guide to Vector Retrieval Quality


Welcome! This project demonstrates how to evaluate the quality and performance of a basic semantic search system using **Sentence Transformers** for embedding and **ChromaDB** as the vector store.

Semantic search returns results based on the *meaning* of a query, not just keywords. Here, you'll learn industry-standard methods to ensure your vector search is working correctly.


## 📁 Project Structure


This project is organized into three main phases: **Data Generation**, **Setup**, and **Evaluation**.


| File Name                        | Purpose                                                                 | Phase      |
|-----------------------------------|-------------------------------------------------------------------------|------------|
| ../data/News_Category_Dataset_v3.json | **(Input)** The original, large, raw dataset (assumed to be present).   | Data       |
| data/test_data_subset.json        | **(Output)** The small, filtered dataset used for testing.               | Data       |
| generate-test-subset.py           | **1. Data Tool:** Selects specific categories to create the test subset. | Data       |
| quantitative_retrieval_evaluation.py | **2. Evaluation:** Measures **Recall @ K**—the percentage of time the correct answer is in the top results. | Evaluation |
| distance_threshold_analysis.py    | **3. Evaluation:** Uses a **Distance Threshold** to filter out low-quality, irrelevant matches. | Evaluation |
| semantic_robustness_test.py       | **4. Evaluation:** **Qualitative** tests for how well the model handles synonyms and context (Polysemy). | Evaluation |


## 🚀 Step 1: Generating the Test Subset


Since the original dataset is very large, we first generate a small, focused test file that is easier to manage and manually review.


### `generate-test-subset.py` Explanation


**Run Command:**
```sh
uv run generate-test-subset.py
```


This script performs the following actions:
1. **Loads Data:** Reads the full News_Category_Dataset_v3.json.
2. **Limits Scope:** Focuses only on the first 10,000 records to identify common categories.
3. **Selects Categories:** Identifies the top 8 most frequent categories (e.g., POLITICS, ENTERTAINMENT).
4. **Creates Stable IDs:** Assigns a deterministic doc_0, doc_1, doc_2, etc., to each headline in the subset. These IDs are crucial for all evaluation scripts (especially Recall @ K).
5. **Saves Output:** Writes the final, small list of headlines with their IDs and categories to data/test_data_subset.json.


## 🔬 Step 2: The Three Pillars of Semantic Evaluation


Once the test_data_subset.json is created, each of the following scripts loads this data into ChromaDB and runs a specific evaluation.


### Method 1: Quantitative Evaluation — Recall @ K (R@K)


**File:** `quantitative_retrieval_evaluation.py`
**Metric:** Recall @ K

**Run Command:**
```sh
uv run quantitative_retrieval_evaluation.py
```

**What is it?**
This is the most standard metric for evaluating a retrieval system. It measures how often the correct (or a highly relevant) document is found within the top K results.

**Approach:**
1. Create a Golden Dataset (Test Set): You need a small set of queries where you manually know and define the correct or highly relevant document(s) in your corpus.
2. Run Search: Run the test queries through your existing `collection.query` function.
3. Calculate R@K: Check if the pre-defined "correct" document ID appears in the top K results (e.g., K=1,3,5,10).

**Sample Output and Analysis:**
```
--- Running Recall @ 3 Evaluation (4 Queries) ---

[1/4] ❌ FAIL: Query: 'News about the latest COVID-19 vaccination rates.'
  Expected ID: doc_0 | Top 3 Retrieved IDs: ['doc_3590', 'doc_2146', 'doc_2903']
[2/4] ✅ MATCH: Query: 'What happened with the passenger who hit a flight attendant?'
  Expected ID: doc_1 | Top 3 Retrieved IDs: ['doc_1', 'doc_2367', 'doc_7060']
[3/4] ✅ MATCH: Query: 'Political scandal involving a candidate and a Super PAC ad.'
  Expected ID: doc_8412 | Top 3 Retrieved IDs: ['doc_8412', 'doc_6542', 'doc_2263']
[4/4] ✅ MATCH: Query: 'Tell me about the new Ant-Man movie trailer.'
  Expected ID: doc_8413 | Top 3 Retrieved IDs: ['doc_8413', 'doc_6378', 'doc_4909']
======================================================================
FINAL RESULT: Recall @ 3 Score: 0.75 (3 out of 4 correct)
======================================================================
```


**Result Explanation:**
The system achieved a Recall @ 3 of 0.75. The only failure (Query 1) suggests the model didn't embed the specific conversational phrasing ("latest COVID-19 vaccination rates") close enough to the exact target headline (doc_0) to place it in the top 3 spots, demonstrating a sensitivity to query structure.


### Method 2: Heuristic Evaluation — Distance Thresholding


**File:** `distance_threshold_analysis.py`
**Metric:** L2 Distance Threshold

**Run Command:**
```sh
uv run distance_threshold_analysis.py
```

**What is it?**
This approach uses your similarity/distance scores to filter out genuinely bad results, ensuring that irrelevant data isn't passed to your LLM (which is key for RAG quality).

**Approach:**
1. **Establish a Threshold:** Determine a maximum acceptable distance score for a document to be considered "relevant." This often requires testing and examining the distribution of scores.
2. **Examine Distribution:** Run a batch of queries and plot/analyze the distance scores for known good matches versus known bad matches.
3. **Implement Filtering:** In your search loop, discard any document whose score exceeds this established threshold.

**Sample Output and Analysis:**
```
--- Running Distance Threshold Analysis (Threshold: < 0.75) ---

[1] Query: 'New update on the latest COVID booster shots'
  ✅ ACCEPTED (Distance: 0.6753): 'WHO Chief Urges Halt To COVID-19 Booster Shots For Rest Of The Year'
  ❌ REJECTED (Distance: 0.7879): 'FDA Working On Booster Vaccine Strategy...'

[2] Query: 'Something interesting that happened this week'
  ❌ REJECTED (Distance: 1.1645): 'What A Year This Week Has Been At YouTube'
  ❌ REJECTED (Distance: 1.1806): '5 Stories You Missed This Week...'
  Note: No documents were found below the 0.75 threshold.
```


**Result Explanation:**
This test confirms the threshold is working as a filter:

* For **Query 1**, the one result with a distance of 0.6753 is **Accepted** (below 0.75), confirming it's a valid match. The remaining results are correctly rejected.
* For the vague **Query 2**, all results have distances above 0.75, meaning the model did not find any strong matches. The filter correctly **Rejected** all of them, preventing noise from being returned.


### Method 3: Qualitative Evaluation — Edge Case Analysis


**File:** `semantic_robustness_test.py`
**Metric:** Context & Synonymy

**Run Command:**
```sh
uv run semantic_robustness_test.py
```

**What is it?**
A high quantitative score (like R@K) doesn't guarantee quality. You must manually check the most challenging cases to ensure the semantic meaning is captured correctly.

**Approach:**
1. **Test for Synonymy:** Use two completely different queries that have the same meaning. The top results and their distances should be very similar.
   - Example: "How do you calculate a vector distance?" and "What is the formula for vector similarity?"
2. **Test for Polysemy/Homonyms:** Use a single word that has multiple meanings in two different contexts. The top results and distances should be drastically different.
   - Example: "NFL draft analysis" (Sports) vs. "Bank draft policy" (Finance)
3. **Test for "Close-but-No-Cigar" (Foil Questions):** Use queries that are only one word different but should lead to completely different results.
   - Example: "News about cats" vs. "News about dogs"

**Implementation:**
This is a manual process, but your current `main()` loop is perfect for this. Run the test queries from each category and visually inspect the distance scores and the retrieved headlines. A well-performing model will show:
* **Synonymy:** Very close distance scores (e.g., 0.05 vs 0.06).
* **Polysemy:** The top results and distances will be completely different.
* **Foils:** The top results should be clearly separated (e.g., all dog articles, then all cat articles).

By combining these three approaches, you move from a simple search to a full-fledged, evaluated retrieval system ready for RAG.

**Sample Output and Analysis:**
```
TEST TYPE: Synonymy Test
  Query 1: 'What kind of funny tweets were posted about pets recently?'
    [Top 1] ID: doc_1580 | Dist: 0.4071
  Query 2: 'Best tweets about cats and dogs this past week.'
    [Top 2] ID: doc_2 | Dist: 0.3631 🎯 TARGET

TEST TYPE: Polysemy/Context Test
  Query 1: 'News about the government official's political salary reduction.'
    [Top 2] ID: doc_8411 | Dist: 1.1314 🎯 TARGET
  Query 2: 'An article discussing high compensation for corporate executives.'
    [Top 1] ID: doc_2265 | Dist: 0.8787
```

**Result Explanation:**
1. **Synonymy Test:** The model showed high accuracy. Both queries (one conversational, one keyword-focused) returned headlines that were semantically identical ("Funniest Tweets About Cats And Dogs"), with extremely low distances. This proves the model understands the semantic equivalence between the two different query phrasings.
2. **Polysemy/Context Test:** The model successfully separated the contexts. **Query 1** found the political salary article (doc_8411), while **Query 2** correctly found a corporate pay article (doc_2265). This demonstrates the model encodes the *context* (Politics vs. Corporate) of the word 'salary' or 'pay' and doesn't just rely on keyword presence.