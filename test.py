def cosine_similarity(a, b):
    return 1 - cosine(a, b)

# Function to convert the string representation to a numpy array
def fingerprint_from_string(fp_str):
    return np.fromstring(fp_str[1:-1], sep=' ')

# Calculate similarity scores
similarity_scores = [cosine_similarity(input_fingerprint, fingerprint_from_string(fp_str)) for fp_str in df['Fingerprint']]

# Add similarity scores to the DataFrame
df['Similarity Score'] = similarity_scores

# Print the results
for index, row in df.iterrows():
    print(f"Person: {row['Person']}, Phrase: {row['Phrase']}, Similarity Score: {row['Similarity Score']}")
