import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist, pos_tag
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

# Download the required NLTK resources
nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Read the Moby Dick file from the Gutenberg dataset
moby_dick = gutenberg.raw('melville-moby_dick.txt')

# Tokenization: Split the entire book into tokens (words)
tokens = word_tokenize(moby_dick)

# Stop-words filtering: Remove stopwords from the tokens
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

# Parts-of-Speech (POS) tagging: Tag the different parts of speech for each word
pos_tags = pos_tag(filtered_tokens)

# POS frequency: Count and display the 5 most common parts of speech and their total counts
tag_fd = FreqDist(tag for (word, tag) in pos_tags)
top_pos = tag_fd.most_common(5)

# Lemmatization: Lemmatize the top 20 tokens
lemmatizer = WordNetLemmatizer()
top_20_tokens = [word for word, _ in FreqDist(filtered_tokens).most_common(20)]
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in top_20_tokens]

# Plotting frequency distribution: Plot a bar chart to visualize the frequency of POS
tag_fd.plot(cumulative=False)

# Perform sentiment analysis (Optional)
# Add your code here to perform sentiment analysis on the text

# Display average sentiment score and overall text sentiment (Optional)
# Add your code here to calculate average sentiment score and determine overall sentiment

# Show the plot
plt.show()
